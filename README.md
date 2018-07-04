# ccminer - With Balloon algo + moany others

Based on Christian Buchner's &amp; Christian H.'s CUDA project
Created by monkins1010 
based on the Fork by tpruvot@github with X14,X15,X17,WHIRL,Blake256 and LYRA2 support , and some others, check the [README.txt](README.txt)
Reforked and optimized by sp-hash@github and KlausT@github 

* KlausT:
    BTC 1QDwdLPrPYSoPqS7pB2kGG84YX6hEcQ4JN
    BCH 1AH1u7B4KtDTUBgmT6NrXyahNEgTac3fL7
* tpruvot:
    BTC donation address: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo
* sp-hash:
    BTC donation address: 1CTiNJyoUmbdMRACtteRWXhGqtSETYd6Vd

A part of the recent algos were originally written by [djm34](https://github.com/djm34).

This variant was tested and built with Visual Studio 2015 on Windows 10

Algos

            balloon     Dreft
			bitcoin     Bitcoin
			blake       Blake 256 (SFR/NEOS)
			blakecoin   Fast Blake 256 (8 rounds)
			c11         X11 variant
			deep        Deepcoin
			dmd-gr      Diamond-Groestl
			fresh       Freshcoin (shavite 80)
			fugue256    Fuguecoin
			groestl     Groestlcoin
			jackpot     Jackpot (JHA)
			keccak      Keccak-256 (Maxcoin)
			luffa       Doomcoin
			lyra2v2     VertCoin
			myr-gr      Myriad-Groestl
			neoscrypt   neoscrypt (FeatherCoin)
			nist5       NIST5 (TalkCoin)
			penta       Pentablake hash (5x Blake 512)
			quark       Quark
			qubit       Qubit
			sia         Siacoin (at pools compatible to siamining.com) 
			skein       Skein SHA2 (Skeincoin)
			s3          S3 (1Coin)
			x11         X11 (DarkCoin)
			x13         X13 (MaruCoin)
			x14         X14
			x15         X15
			x17         X17 (peoplecurrency)
			vanilla     Blake 256 8 rounds
			whirl       Whirlcoin (old whirlpool)
			whirlpoolx  Vanillacoin 

## Source code dependencies

* Jansson
* OpenSSL (prebuilt for win)
* Curl (prebuilt for win)
* pthreads (prebuilt for win)

This fork contains these libraries for x64 Windows.
