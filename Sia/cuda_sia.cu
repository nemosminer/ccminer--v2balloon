#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <cuda_helper.h>
#include "balloon.h"
#include "../sha256-sse/sha256.h"
#include "miner.h"
#include "../neoscrypt/cuda_vector_tpruvot.cuh"

static uint32_t *d_KNonce2[MAX_GPUS];
__global__ void conv_onethread(int n, int fn, const float * signal, const float * filter, float * retSignal);
__device__ void cuda_hash_state_mix(struct hash_state *s, int32_t mixrounds, uint64_t *prebuf_le);
__device__ void device_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx);
__device__ void device_sha256_168byte(uint8_t *data, uint8_t *outhash);
__device__ void cuda_hash_state_extract(const struct hash_state *s, uint8_t out[BLOCK_SIZE]);
__device__ void cuda_compress(uint64_t *counter, uint8_t *out, const uint8_t *blocks[], size_t blocks_to_comp);
__device__ void cuda_expand(uint64_t *counter, uint8_t *buf, size_t blocks_in_buf);
__device__ void cuda_hash_state_fill(struct hash_state *s, const uint8_t *in, size_t inlen, int32_t t_cost, int64_t s_cost);
__device__ void device_sha256_generic(uint8_t *data, uint8_t *outhash, uint32_t len);
void host_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx);
__global__ void cudaized_multi(struct hash_state *s, int32_t mixrounds, uint64_t *prebuf_le, uint8_t *input, uint32_t len, uint8_t *output, int64_t s_cost, uint32_t max_nonce, int thr_id, uint32_t *winning_nonce, uint32_t num_threads, uint32_t *device_target, uint32_t *is_winning, uint32_t num_blocks, uint8_t *sbufs, uint32_t *d_KNonce2);
void update_device_data(int thr_id);

//#define DEBUG
//#define CUDA_DEBUG
//#define CUDA_OUTPUT

//#define DEBUG
//#define DEBUG_CUDA
//#define LOWMEM

int cuda_query() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}

	return nDevices;
}

__constant__ const uint32_t __align__(8) __sha256_init[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};
__constant__ const uint32_t cpu_K[64] = {
	0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U, 0xAB1C5ED5U,
	0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U,
	0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU,
	0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U,
	0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U,
	0xA2BFE8A1U, 0xA81A664BU, 0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U,
	0x19A4C116U, 0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
	0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U
};


#define PREBUF_LEN 409600
uint64_t host_prebuf_le[20][PREBUF_LEN / 8];
uint8_t host_prebuf_filled[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
#define BLOCK_SIZE (32)

uint64_t *device_prebuf_le[20];
uint32_t *device_winning_nonce[20];
uint8_t *device_sbuf[20];
struct hash_state *device_s[20];
uint32_t *device_target[20];
uint32_t *device_is_winning[20];
uint8_t *device_out[20];
uint8_t *device_input[20];
uint8_t *device_sbufs[20];

uint8_t balloon_inited[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
uint8_t syncmode_set[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
void balloon_cuda_init(int thr_id, uint32_t opt_cuda_syncmode, uint32_t num_threads, uint32_t num_blocks) {
	CUDA_SAFE_CALL(cudaSetDevice(thr_id));
	
	if (!syncmode_set[thr_id]) {
		//CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		//CUDA_SAFE_CALL(cudaDeviceReset());
		//CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
		//CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		//CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		//CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_K, K, sizeof(K), 0, cudaMemcpyHostToDevice));
		//cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
		syncmode_set[thr_id] = 1;
	}
#ifdef DEBUG
	printf("DEBUG GPU %d: entering balloon_cuda_init\n", thr_id);
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	printf("%d KB free of total %d KB before init\n", free / 1024, total / 1024);
#endif
	if (!balloon_inited[thr_id]) {
		applog(LOG_INFO, "Initiated GPU #%d", thr_id);
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_prebuf_le[thr_id], (PREBUF_LEN / 8) * sizeof(uint64_t)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_sbuf[thr_id], /*s.n_blocks*/4096 * BLOCK_SIZE));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_is_winning[thr_id], sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_winning_nonce[thr_id], sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_s[thr_id], sizeof(struct hash_state)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_target[thr_id], 8 * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_out[thr_id], BLOCK_SIZE * sizeof(uint8_t)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_input[thr_id], /*len*/80));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_KNonce2[thr_id], 2 * sizeof(uint32_t)));

		
#ifdef LOWMEM
		CUDA_SAFE_CALL(cudaMalloc((void**)&device_sbufs[thr_id], num_threads*num_blocks * 4096 * BLOCK_SIZE));
		printf("device_sbufs[thr_id] = %x\n", device_sbufs[thr_id]);
#endif
		balloon_inited[thr_id] = 1;
	}
#ifdef DEBUG
	printf("DEBUG GPU %d: leaving balloon_cuda_init\n", thr_id);
	cudaMemGetInfo(&free, &total);
	printf("%d KB free of total %d KB after init\n", free / 1024, total / 1024);
#endif

}

void fill_prebuf(struct hash_state *s, int thr_id) {
#ifdef DEBUG
	printf("DEBUG GPU %d: entering fill_prebuf\n", thr_id);
#endif
	uint8_t host_prebuf[PREBUF_LEN];
	if (!host_prebuf_filled[thr_id]) {
		bitstream_fill_buffer(&s->bstream, host_prebuf, PREBUF_LEN);
		host_prebuf_filled[thr_id] = 1;
		uint8_t *buf = host_prebuf;
		uint64_t *lebuf = host_prebuf_le[thr_id];
		for (int i = 0; i < PREBUF_LEN; i += 8) {
			//bytes_to_littleend8_uint64(buf, lebuf);
			*lebuf <<= 8; *lebuf |= *(buf + 7);
			*lebuf <<= 8; *lebuf |= *(buf + 6);
			*lebuf <<= 8; *lebuf |= *(buf + 5);
			*lebuf <<= 8; *lebuf |= *(buf + 4);
			*lebuf <<= 8; *lebuf |= *(buf + 3);
			*lebuf <<= 8; *lebuf |= *(buf + 2);
			*lebuf <<= 8; *lebuf |= *(buf + 1);
			*lebuf <<= 8; *lebuf |= *(buf + 0);
			*lebuf %= 4096;
			*lebuf <<= 5;
			lebuf++;
			buf += 8;
		}
		update_device_data(thr_id);
	}
#ifdef DEBUG
	printf("DEBUG GPU %d: leaving fill_prebuf\n", thr_id);
#endif
}

void reset_host_prebuf(int thr_id) {
	
		host_prebuf_filled[thr_id] = 0;
	
}


void update_device_data(int thr_id) {
#ifdef DEBUG
	printf("DEBUG GPU %d: entering update_device_data\n", thr_id);
#endif
	CUDA_SAFE_CALL(cudaMemcpy(device_prebuf_le[thr_id], host_prebuf_le[thr_id], (PREBUF_LEN / 8) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpyToSymbol(device_prebuf_le, host_prebuf_le, 409600/8 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
#ifdef DEBUG
	printf("DEBUG GPU %d: leaving update_device_data\n", thr_id);
#endif
}

void balloon_cuda_free(int thr_id) {
	//cudaFree(device_prebuf_le[thr_id]);
	//cudaFree(device_sbuf[thr_id]);
	//cudaFree(device_s[thr_id]);
	//cudaFree(device_winning_nonce[thr_id]);
	//cudaFree(device_is_winning[thr_id]);
	//cudaFree(device_out[thr_id]);
	//cudaFree(device_input[thr_id]);
#ifdef LOWMEM
	//cudaFree(device_sbufs[thr_id]);
#endif
	//balloon_inited = 0;
}

uint32_t balloon_128_cuda(int thr_id, unsigned char *input, unsigned char *output, uint32_t *target, uint32_t max_nonce, uint32_t num_threads, uint32_t *is_winning, uint32_t num_blocks, uint32_t *h_nounce) {
	return cuda_balloon(thr_id, input, output, 80, 128, 4, target, max_nonce, num_threads, is_winning, num_blocks,h_nounce);
}

//#define NUM_THREADS 256
//#define NUM_THREADS 384
//#define NUM_THREADS 384
//#define NUM_BLOCKS 480
//#define NUM_BLOCKS 48

__host__
uint32_t cuda_balloon(int thr_id, unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost, uint32_t *target, uint32_t max_nonce, uint32_t num_threads, uint32_t *ret_is_winning, uint32_t num_blocks, uint32_t *h_nounce) {
#ifdef DEBUG
	printf("DEBUG GPU %d: entering cuda_balloon\n", thr_id);
#endif

	CUDA_SAFE_CALL(cudaSetDevice(thr_id));
	struct balloon_options opts;
	struct hash_state s;
	balloon_init(&opts, s_cost, t_cost);
	hash_state_init(&s, &opts, input);
	fill_prebuf(&s, thr_id);
	uint8_t *pc_sbuf = s.buffer;

#ifdef DEBUG
	if (s.n_blocks > 4096) printf("s.n_blocks = %llu\n", s.n_blocks);
#endif

	uint32_t first_nonce = ((input[76] << 24) | (input[77] << 16) | (input[78] << 8) | input[79]);
	CUDA_SAFE_CALL(cudaMemcpy((void**)device_sbuf[thr_id], (void**)s.buffer, s.n_blocks * BLOCK_SIZE, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemsetAsync((void**)d_KNonce2[thr_id], 0xff, 2 * sizeof(uint32_t), gpustream[thr_id]));
	s.buffer = device_sbuf[thr_id];
	CUDA_SAFE_CALL(cudaMemcpy((void**)device_s[thr_id], (void**)&s, sizeof(struct hash_state), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void**)device_input[thr_id], (void**)input, len, cudaMemcpyHostToDevice));
	uint32_t host_winning_nonce = 0;
	uint32_t host_is_winning = 0;
	CUDA_SAFE_CALL(cudaMemcpy(device_target[thr_id], target, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void**)device_winning_nonce[thr_id], (void**)&host_winning_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void**)device_is_winning[thr_id], (void**)&host_is_winning, sizeof(uint32_t), cudaMemcpyHostToDevice));
	cudaized_multi << <num_blocks, num_threads, 0, gpustream[thr_id] >> > (device_s[thr_id], t_cost, device_prebuf_le[thr_id], device_input[thr_id], len, device_out[thr_id], s_cost, max_nonce, thr_id, device_winning_nonce[thr_id], num_threads, device_target[thr_id], device_is_winning[thr_id], num_blocks, device_sbufs[thr_id], d_KNonce2[thr_id]);
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy((void*)&host_winning_nonce, (void*)device_winning_nonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy((void*)&host_is_winning, (void*)device_is_winning[thr_id],sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy((void**)h_nounce, (void**)d_KNonce2[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

#ifdef DEBUG
	if (host_is_winning) {
		printf("[Host (GPU %d)] Winning (%d) nonce: %u\n", thr_id, host_is_winning, host_winning_nonce);
	}
#endif

#ifdef CUDA_OUTPUT
	CUDA_SAFE_CALL(cudaMemcpy((void**)output, (void**)device_out[thr_id], BLOCK_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost));
#endif



	s.buffer = pc_sbuf;
	hash_state_free(&s);

	*ret_is_winning = host_is_winning;
	if (host_is_winning == 0) {
		host_winning_nonce = first_nonce + num_threads*num_blocks - 1;

	}

	return host_winning_nonce;
}

__device__ void * block_index(const struct hash_state *s, size_t i) {
	return s->buffer + (BLOCK_SIZE * i);
}
__device__ void * block_last(const struct hash_state *s) {
	return block_index(s, s->n_blocks - 1);
}
__device__ void cuda_hash_state_extract(const struct hash_state *s, uint8_t out[BLOCK_SIZE]) {
	uint8_t *b = (uint8_t*)block_last(s);
	memcpy((char *)out, (const char *)b, BLOCK_SIZE);
}

//#define CUDA_OUTPUT
__global__ void cudaized_multi(struct hash_state *hs, int32_t mixrounds, uint64_t *prebuf_le, uint8_t *input, uint32_t len, uint8_t *output, int64_t s_cost, uint32_t max_nonce, int thr_id, uint32_t *winning_nonce, uint32_t num_threads, uint32_t *device_target, uint32_t *is_winning, uint32_t num_blocks, uint8_t *sbufs, uint32_t *resNounce) {

	//if (threadIdx.x < 64U) s_K[threadIdx.x] = c_K[threadIdx.x];

#ifdef DEBUG_CUDA
	printf("[Device %d] entering cuda\n", thr_id);
#endif
	uint32_t id = blockDim.x*blockIdx.x + threadIdx.x;
	uint32_t nonce = ((input[76] << 24) | (input[77] << 16) | (input[78] << 8) | input[79]) + id;
	//if (nonce > max_nonce || *is_winning) {
	if (nonce > max_nonce || *is_winning) {

		asm("exit;");
	}
	uint8_t local_input[80];
#ifdef CUDA_OUTPUT
	uint8_t local_output[32];
#endif
	struct hash_state local_s;
	memcpy(local_input, input, len);
	memcpy(&local_s, hs, sizeof(struct hash_state));

#ifdef LOWMEM
	uint8_t *local_sbuf = sbufs + id * 4096 * BLOCK_SIZE;
#else
	uint8_t local_sbuf[4096 * BLOCK_SIZE];
#endif

#ifdef LOWMEM
	memcpy(local_sbuf, hs->buffer, 4096 * BLOCK_SIZE);
#else
	memcpy(&local_sbuf, hs->buffer, 4096 * BLOCK_SIZE);
#endif

	local_s.buffer = local_sbuf;
	((uint32_t*)local_input)[19] = ((nonce & 0xff000000) >> 24) | ((nonce & 0xff0000) >> 8) | ((nonce & 0xff00) << 8) | ((nonce & 0xff) << 24);
	local_s.counter = 0;
	cuda_hash_state_fill(&local_s, local_input, len, mixrounds, s_cost);
	cuda_hash_state_mix(&local_s, mixrounds, prebuf_le);

	if (((uint32_t*)(local_sbuf + (4095 << 5)))[7] < device_target[7]) {

		// Assume winning nonce

		*winning_nonce = nonce;
		*is_winning = 1;

		__threadfence_system();
		asm("exit;");
	}

	}


__device__ void cuda_expand(uint64_t *counter, uint8_t *buf, size_t blocks_in_buf) {
	const uint8_t *blocks[1] = { buf };
	uint8_t *cur = buf + BLOCK_SIZE;

	for (size_t i = 1; i < blocks_in_buf; i++) {
		cuda_compress(counter, cur, blocks,1);
		*blocks += BLOCK_SIZE;
		cur += BLOCK_SIZE;
	}
}



__device__ void cuda_compress(uint64_t *counter, uint8_t *out, const uint8_t *blocks[], size_t blocks_to_comp) {
	uint8_t data[168];
	uint8_t *dp = (uint8_t*)data;
	uint8_t len = BLOCK_SIZE * blocks_to_comp + 8;
	memcpy(dp, counter, 8);
	dp += 8;
	for (unsigned int i = 0; i < blocks_to_comp; i++) {
		memcpy(dp, *(blocks + i), BLOCK_SIZE);
		dp += BLOCK_SIZE;
	}
	device_sha256_generic(data, out, len);
	*counter += 1;
}

__device__ void cuda_hash_state_fill(struct hash_state *s, const uint8_t *in, size_t inlen, int32_t t_cost, int64_t s_cost) {

	
	uint8_t data[132];
	//uint32_t shalen = 8+SALT_LEN+inlen+8+4;
	uint8_t *dp = (uint8_t*)data;
	//if (inlen != 80) {
	//	printf("inlen != 128 (inlen = %d)!!\n", inlen);
	//	if (inlen > 80) inlen = 80;
	//}
	memcpy(dp, &s->counter, 8);
	dp += 8;
	memcpy(dp, in, SALT_LEN);
	dp += SALT_LEN;
	memcpy(dp, in, inlen);
	dp += inlen;
	memcpy(dp, &s_cost, 8);
	dp += 8;
	memcpy(dp, &t_cost, 4);

	device_sha256_generic(data, s->buffer, 132);
	s->counter++;
	cuda_expand(&s->counter, s->buffer, s->n_blocks);
}




__device__ void cuda_hash_state_mix(struct hash_state *s, int32_t mixrounds, uint64_t *prebuf_le) {
	

	
	uint64_t *buf = prebuf_le;
	uint8_t *sbuf = s->buffer;
	const int32_t n_blocks = 4096;
	mixrounds = 4;
	uint8_t *last_block = (sbuf + (BLOCK_SIZE*(n_blocks - 1)));
	uint8_t *blocks[5];
	unsigned char data[8 + BLOCK_SIZE * 5];
	unsigned char *db1 = data + 8;
	unsigned char *db2 = data + 40;
	unsigned char *db3 = data + 72;
	unsigned char *db4 = data + 104;
	unsigned char *db5 = data + 136;
#pragma unroll
	for (int32_t rounds = 0; rounds < mixrounds; rounds++) {
		{ // i = 0
			blocks[0] = last_block;
			blocks[1] = sbuf;
			blocks[2] = (sbuf + ((*(buf++))));
			blocks[3] = (sbuf + ((*(buf++))));
			blocks[4] = (sbuf + ((*(buf++))));

			// New sha256
			//block = (uint8_t**)blocks;
			memcpy(data, &s->counter, 8);
			memcpy(db1, blocks[0], BLOCK_SIZE);
			memcpy(db2, blocks[1], BLOCK_SIZE);
			memcpy(db3, blocks[2], BLOCK_SIZE);
			memcpy(db4, blocks[3], BLOCK_SIZE);
			memcpy(db5, blocks[4], BLOCK_SIZE);
			device_sha256_168byte(data, (uint8_t*)blocks[1]);
			s->counter++;
		}
#pragma unroll
		for (size_t i = 1; i < n_blocks; i++) {
			blocks[0] = blocks[1];
			blocks[1] += BLOCK_SIZE;
			/*blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));*/

			blocks[2] = (sbuf + ((*(buf++))));
			blocks[3] = (sbuf + ((*(buf++))));
			blocks[4] = (sbuf + ((*(buf++))));

			// New sha256
			memcpy(data, &s->counter, 8);
			memcpy(db1, blocks[0], BLOCK_SIZE);
			memcpy(db2, blocks[1], BLOCK_SIZE);
			memcpy(db3, blocks[2], BLOCK_SIZE);
			memcpy(db4, blocks[3], BLOCK_SIZE);
			memcpy(db5, blocks[4], BLOCK_SIZE);
			device_sha256_168byte(data, (uint8_t*)blocks[1]);
			s->counter++;
		}
		//s->has_mixed = true;
	}
#ifdef DEBUG_CUDA
	if (buf - prebuf_le > 49152) printf("prebuf_le max used: %d, mixrounds = %d, n_blocks = %d\n", buf - prebuf_le, mixrounds, n_blocks);
#endif
}

/*
#define SHA256_CONST(x)         (SHA256_CONST_ ## x)

// constants, as provided in FIPS 180-2 

#define SHA256_CONST_0          0x428a2f98U
#define SHA256_CONST_1          0x71374491U
#define SHA256_CONST_2          0xb5c0fbcfU
#define SHA256_CONST_3          0xe9b5dba5U
#define SHA256_CONST_4          0x3956c25bU
#define SHA256_CONST_5          0x59f111f1U
#define SHA256_CONST_6          0x923f82a4U
#define SHA256_CONST_7          0xab1c5ed5U

#define SHA256_CONST_8          0xd807aa98U
#define SHA256_CONST_9          0x12835b01U
#define SHA256_CONST_10         0x243185beU
#define SHA256_CONST_11         0x550c7dc3U
#define SHA256_CONST_12         0x72be5d74U
#define SHA256_CONST_13         0x80deb1feU
#define SHA256_CONST_14         0x9bdc06a7U
#define SHA256_CONST_15         0xc19bf174U

#define SHA256_CONST_16         0xe49b69c1U
#define SHA256_CONST_17         0xefbe4786U
#define SHA256_CONST_18         0x0fc19dc6U
#define SHA256_CONST_19         0x240ca1ccU
#define SHA256_CONST_20         0x2de92c6fU
#define SHA256_CONST_21         0x4a7484aaU
#define SHA256_CONST_22         0x5cb0a9dcU
#define SHA256_CONST_23         0x76f988daU

#define SHA256_CONST_24         0x983e5152U
#define SHA256_CONST_25         0xa831c66dU
#define SHA256_CONST_26         0xb00327c8U
#define SHA256_CONST_27         0xbf597fc7U
#define SHA256_CONST_28         0xc6e00bf3U
#define SHA256_CONST_29         0xd5a79147U
#define SHA256_CONST_30         0x06ca6351U
#define SHA256_CONST_31         0x14292967U

#define SHA256_CONST_32         0x27b70a85U
#define SHA256_CONST_33         0x2e1b2138U
#define SHA256_CONST_34         0x4d2c6dfcU
#define SHA256_CONST_35         0x53380d13U
#define SHA256_CONST_36         0x650a7354U
#define SHA256_CONST_37         0x766a0abbU
#define SHA256_CONST_38         0x81c2c92eU
#define SHA256_CONST_39         0x92722c85U

#define SHA256_CONST_40         0xa2bfe8a1U
#define SHA256_CONST_41         0xa81a664bU
#define SHA256_CONST_42         0xc24b8b70U
#define SHA256_CONST_43         0xc76c51a3U
#define SHA256_CONST_44         0xd192e819U
#define SHA256_CONST_45         0xd6990624U
#define SHA256_CONST_46         0xf40e3585U
#define SHA256_CONST_47         0x106aa070U

#define SHA256_CONST_48         0x19a4c116U
#define SHA256_CONST_49         0x1e376c08U
#define SHA256_CONST_50         0x2748774cU
#define SHA256_CONST_51         0x34b0bcb5U
#define SHA256_CONST_52         0x391c0cb3U
#define SHA256_CONST_53         0x4ed8aa4aU
#define SHA256_CONST_54         0x5b9cca4fU
#define SHA256_CONST_55         0x682e6ff3U

#define SHA256_CONST_56         0x748f82eeU
#define SHA256_CONST_57         0x78a5636fU
#define SHA256_CONST_58         0x84c87814U
#define SHA256_CONST_59         0x8cc70208U
#define SHA256_CONST_60         0x90befffaU
#define SHA256_CONST_61         0xa4506cebU
#define SHA256_CONST_62         0xbef9a3f7U
#define SHA256_CONST_63         0xc67178f2U
*/

/* Ch and Maj are the basic SHA2 functions. */
#define Ch(b, c, d)     (((b) & (c)) ^ ((~b) & (d)))
#define Maj(b, c, d)    (((b) & (c)) ^ ((b) & (d)) ^ ((c) & (d)))

/* Rotates x right n bits. */
#define ROTR(x, n) __funnelshift_r( (x), (x), (n) )

/* Shift x right n bits */
#define SHR(x, n)       ((x) >> (n))

/* SHA256 Functions */
#define BIGSIGMA0_256(x)        (ROTR((x), 2) ^ ROTR((x), 13) ^ ROTR((x), 22))
#define BIGSIGMA1_256(x)        (ROTR((x), 6) ^ ROTR((x), 11) ^ ROTR((x), 25))
#define SIGMA0_256(x)           (ROTR((x), 7) ^ ROTR((x), 18) ^ SHR((x), 3))
#define SIGMA1_256(x)           (ROTR((x), 17) ^ ROTR((x), 19) ^ SHR((x), 10))

#define	SHA256ROUND(a, b, c, d, e, f, g, h, i, w)			\
T1 = h + BIGSIGMA1_256(e) + Ch(e, f, g) + cpu_K[(i)] + w;	\
d += T1;							\
T2 = BIGSIGMA0_256(a) + Maj(a, b, c);				\
h = T1 + T2

#define        LOAD_BIG_32(addr) (((addr)[0] << 24) | ((addr)[1] << 16) | ((addr)[2] << 8) | (addr)[3])


__device__ void device_sha256_168byte(uint8_t *data, uint8_t *outhash) {
	 __sha256_block_t block[3];
	uint8_t *ptr = (uint8_t*)block;
	memcpy(ptr, data, 168);
	ptr += 168;
	*ptr++ = 0x80;
	memset(ptr, 0, 21);
	ptr += 21;
	*ptr++ = 0x5;
	*ptr++ = 0x40;
	 __sha256_hash_t ohash;
	memcpy(ohash, __sha256_init, 32);
	uint32_t a = ohash[0];
	uint32_t b = ohash[1];
	uint32_t c = ohash[2];
	uint32_t d = ohash[3];
	uint32_t e = ohash[4];
	uint32_t f = ohash[5];
	uint32_t g = ohash[6];
	uint32_t h = ohash[7];
	register uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	register uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
	register uint32_t T1, T2;
	w0 = LOAD_BIG_32(block[0] + 4 * 0);  SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
	w1 = LOAD_BIG_32(block[0] + 4 * 1);  SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	w2 = LOAD_BIG_32(block[0] + 4 * 2);  SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	w3 = LOAD_BIG_32(block[0] + 4 * 3);  SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	w4 = LOAD_BIG_32(block[0] + 4 * 4);  SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	w5 = LOAD_BIG_32(block[0] + 4 * 5);  SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	w6 = LOAD_BIG_32(block[0] + 4 * 6);  SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	w7 = LOAD_BIG_32(block[0] + 4 * 7);  SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	w8 = LOAD_BIG_32(block[0] + 4 * 8);  SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	w9 = LOAD_BIG_32(block[0] + 4 * 9);  SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	w10 = LOAD_BIG_32(block[0] + 4 * 10); SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	w11 = LOAD_BIG_32(block[0] + 4 * 11); SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	w12 = LOAD_BIG_32(block[0] + 4 * 12); SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	w13 = LOAD_BIG_32(block[0] + 4 * 13); SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	w14 = LOAD_BIG_32(block[0] + 4 * 14); SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	w15 = LOAD_BIG_32(block[0] + 4 * 15); SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);
	a = (ohash[0] += a);
	b = (ohash[1] += b);
	c = (ohash[2] += c);
	d = (ohash[3] += d);
	e = (ohash[4] += e);
	f = (ohash[5] += f);
	g = (ohash[6] += g);
	h = (ohash[7] += h);
	w0 = LOAD_BIG_32(block[1] + 4 * 0);  SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
	w1 = LOAD_BIG_32(block[1] + 4 * 1);  SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	w2 = LOAD_BIG_32(block[1] + 4 * 2);  SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	w3 = LOAD_BIG_32(block[1] + 4 * 3);  SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	w4 = LOAD_BIG_32(block[1] + 4 * 4);  SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	w5 = LOAD_BIG_32(block[1] + 4 * 5);  SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	w6 = LOAD_BIG_32(block[1] + 4 * 6);  SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	w7 = LOAD_BIG_32(block[1] + 4 * 7);  SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	w8 = LOAD_BIG_32(block[1] + 4 * 8);  SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	w9 = LOAD_BIG_32(block[1] + 4 * 9);  SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	w10 = LOAD_BIG_32(block[1] + 4 * 10); SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	w11 = LOAD_BIG_32(block[1] + 4 * 11); SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	w12 = LOAD_BIG_32(block[1] + 4 * 12); SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	w13 = LOAD_BIG_32(block[1] + 4 * 13); SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	w14 = LOAD_BIG_32(block[1] + 4 * 14); SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	w15 = LOAD_BIG_32(block[1] + 4 * 15); SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);
	a = (ohash[0] += a);
	b = (ohash[1] += b);
	c = (ohash[2] += c);
	d = (ohash[3] += d);
	e = (ohash[4] += e);
	f = (ohash[5] += f);
	g = (ohash[6] += g);
	h = (ohash[7] += h);
	w0 = LOAD_BIG_32(block[2] + 4 * 0);  SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
	w1 = LOAD_BIG_32(block[2] + 4 * 1);  SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	w2 = LOAD_BIG_32(block[2] + 4 * 2);  SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	w3 = LOAD_BIG_32(block[2] + 4 * 3);  SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	w4 = LOAD_BIG_32(block[2] + 4 * 4);  SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	w5 = LOAD_BIG_32(block[2] + 4 * 5);  SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	w6 = LOAD_BIG_32(block[2] + 4 * 6);  SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	w7 = LOAD_BIG_32(block[2] + 4 * 7);  SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	w8 = LOAD_BIG_32(block[2] + 4 * 8);  SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	w9 = LOAD_BIG_32(block[2] + 4 * 9);  SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	w10 = LOAD_BIG_32(block[2] + 4 * 10); SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	w11 = LOAD_BIG_32(block[2] + 4 * 11); SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	w12 = LOAD_BIG_32(block[2] + 4 * 12); SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	w13 = LOAD_BIG_32(block[2] + 4 * 13); SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	w14 = LOAD_BIG_32(block[2] + 4 * 14); SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	w15 = LOAD_BIG_32(block[2] + 4 * 15); SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);
	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);
	ohash[0] += a;
	ohash[1] += b;
	ohash[2] += c;
	ohash[3] += d;
	ohash[4] += e;
	ohash[5] += f;
	ohash[6] += g;
	ohash[7] += h;
	// finmessy
	uint8_t *h2 = (uint8_t*)ohash;
	uint8_t *outp2 = outhash;
#pragma unroll
	for (int i = 0; i < 32 / 4; i++) {
		// Fix endianness at the same time
		*outp2++ = h2[3];
		*outp2++ = h2[2];
		*outp2++ = h2[1];
		*outp2++ = h2[0];
		h2 += 4;
	}
}

__device__ void device_sha256_generic(uint8_t *data, uint8_t *outhash, uint32_t len) {
	if (len > 184) {
		printf("Longer than 3 blocks (184bytes), sha256_generic not made for this..\n");
		len = 184;
	}
	uint8_t num_blocks = len / 64 + 1;
	uint32_t tot_len = num_blocks * 512 - 65;
	uint32_t num_padding = (tot_len - len * 8) / 8;
	__sha256_block_t block[3];
	uint8_t *ptr = (uint8_t*)block;
	memcpy(ptr, data, len);
	ptr += len;
	*ptr++ = 0x80;
	memset(ptr, 0, num_padding);
	ptr += num_padding;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = ((len * 8) & 0xff00) >> 8;
	*ptr++ = (len * 8) & 0xff;
	__sha256_hash_t ohash;
	memcpy(ohash, __sha256_init, 32);
#pragma unroll
	for (int i = 0; i < num_blocks; i++) {
		uint32_t a = ohash[0];
		uint32_t b = ohash[1];
		uint32_t c = ohash[2];
		uint32_t d = ohash[3];
		uint32_t e = ohash[4];
		uint32_t f = ohash[5];
		uint32_t g = ohash[6];
		uint32_t h = ohash[7];
		uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
		uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
		uint32_t T1, T2;
		w0 = LOAD_BIG_32(block[i] + 4 * 0);  SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
		w1 = LOAD_BIG_32(block[i] + 4 * 1);  SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
		w2 = LOAD_BIG_32(block[i] + 4 * 2);  SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
		w3 = LOAD_BIG_32(block[i] + 4 * 3);  SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
		w4 = LOAD_BIG_32(block[i] + 4 * 4);  SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
		w5 = LOAD_BIG_32(block[i] + 4 * 5);  SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
		w6 = LOAD_BIG_32(block[i] + 4 * 6);  SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
		w7 = LOAD_BIG_32(block[i] + 4 * 7);  SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
		w8 = LOAD_BIG_32(block[i] + 4 * 8);  SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
		w9 = LOAD_BIG_32(block[i] + 4 * 9);  SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
		w10 = LOAD_BIG_32(block[i] + 4 * 10); SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
		w11 = LOAD_BIG_32(block[i] + 4 * 11); SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
		w12 = LOAD_BIG_32(block[i] + 4 * 12); SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
		w13 = LOAD_BIG_32(block[i] + 4 * 13); SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
		w14 = LOAD_BIG_32(block[i] + 4 * 14); SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
		w15 = LOAD_BIG_32(block[i] + 4 * 15); SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);
		w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
		w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
		w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
		w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
		w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
		w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
		w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
		w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
		w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
		w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
		w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
		w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
		w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
		w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
		w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
		w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);
		w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
		w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
		w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
		w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
		w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
		w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
		w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
		w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
		w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
		w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
		w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
		w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
		w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
		w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
		w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
		w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);
		w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0; SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
		w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1; SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
		w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2; SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
		w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3; SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
		w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4; SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
		w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5; SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
		w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6; SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
		w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7; SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
		w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8; SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
		w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9; SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
		w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10; SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
		w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11; SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
		w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12; SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
		w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13; SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
		w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14; SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
		w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15; SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);
		ohash[0] += a;
		ohash[1] += b;
		ohash[2] += c;
		ohash[3] += d;
		ohash[4] += e;
		ohash[5] += f;
		ohash[6] += g;
		ohash[7] += h;
	}

	uint8_t *h = (uint8_t*)ohash;
	uint8_t *outp = outhash;
#pragma unroll
	for (int i = 0; i < 32 / 4; i++) {
		*outp++ = h[3];
		*outp++ = h[2];
		*outp++ = h[1];
		*outp++ = h[0];
		h += 4;
	}
}

