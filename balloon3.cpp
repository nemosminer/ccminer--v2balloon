/*
* balloon algorithm
*
*/
#include "miner.h"
#include <string.h>
#include <stdint.h>

#include <openssl/sha.h>

#include "balloon.h"
#include "cuda_helper.h"
uint32_t prev_pdata[20][10];
uint32_t num_cuda_threads = 64;
uint32_t num_cuda_blocks = 48;
extern void balloon_gpu_init(int thr_id);
uint32_t balloon_cpu_hash(int thr_id, unsigned char *input, uint32_t threads, uint32_t startNounce, uint32_t *h_nounce, uint32_t max_nonce);
extern void balloon_setBlock_80(int thr_id, void *pdata, const void *ptarget);

//int scanhash_balloon(int thr_id, struct work *work, uint32_t max_nonce, uint32_t *hashes_done)
int scanhash_balloon(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	uint32_t _ALIGN(128) hash32[8];
	uint32_t _ALIGN(128) cudahash32[8];
	uint32_t _ALIGN(128) verifyhash32[8];
	uint32_t _ALIGN(128) endiandata[20];
	
	uint8_t pdata_changed = 0;
	for (int i = 0; i < 10; i++) {
		if (prev_pdata[thr_id][i] != pdata[i]) {
			prev_pdata[thr_id][i] = pdata[i];
			pdata_changed = 1;
		}
	}

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce;

	for (int i = 0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	};

	
	static THREAD volatile bool init = false;
	if (!init)
	{
		//CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		//CUDA_SAFE_CALL(cudaDeviceReset());
		//CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		//CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		//CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		
		mining_has_stopped[thr_id] = false;
		init = true;
	}
	
	balloon_cuda_init(thr_id, 0, num_cuda_threads, num_cuda_blocks);
	if (pdata_changed) {
		balloon_reset();
		reset_host_prebuf();
	}
	do {
		be32enc(&endiandata[19], n);
		
		uint32_t is_winning = 0;
		uint32_t winning_nonce = balloon_128_cuda(thr_id, (unsigned char *)endiandata, (unsigned char*)cudahash32, ptarget, max_nonce, num_cuda_threads, &is_winning, num_cuda_blocks);
		if (stop_mining) { mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr); }
		be32enc(&endiandata[19], n);
		n = winning_nonce;
		

		if (is_winning) {
			balloon_128_orig((unsigned char *)endiandata, (unsigned char*)verifyhash32);
			memcpy(hash32, verifyhash32, 32);

			//memcpy(hash32, cudahash32, 32);

		}

		//balloon_128((unsigned char *)endiandata, (unsigned char *)hash32);

		if (is_winning ) {
			//applog(LOG_INFO, "Submitting nonce %08x on gpu %u\n", winning_nonce, thr_id);

			//work_set_target_ratio(work, hash32);
			*hashes_done = n - first_nonce + 1;
			pdata[19] = n;
			//balloon_cuda_free(thr_id);
			return 1;
		}
		n++;
	CUDA_SAFE_CALL(cudaGetLastError());
	} while (n < max_nonce && !work_restart[thr_id].restart);
    //balloon_cuda_free(thr_id);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;

	return 0;
}