// Copyright (c) 2012-2013 The Cryptonote developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common/int-util.h"
#include "hash-ops.h"
#include "oaes_lib.h"

#include <emmintrin.h>

#ifdef __unix__
#include <sys/mman.h>
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <intrin.h>
#define STATIC
#define INLINE __inline
#if !defined(RDATA_ALIGN16)
#define RDATA_ALIGN16 __declspec(align(16))
#endif
#else
#include <wmmintrin.h>
#define STATIC static
#define INLINE inline
#if !defined(RDATA_ALIGN16)
#define RDATA_ALIGN16 __attribute__ ((aligned(16)))
#endif
#endif

#define MEMORY         (1 << 21) // 2MB scratchpad
#define ITER           (1 << 20)
#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    32
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE (INIT_SIZE_BLK * AES_BLOCK_SIZE)

#define U64(x) ((uint64_t *) (x))
#define R128(x) ((__m128i *) (x))

extern int aesb_single_round(const uint8_t *in, uint8_t*out, const uint8_t *expandedKey);
extern int aesb_pseudo_round(const uint8_t *in, uint8_t *out, const uint8_t *expandedKey);

#pragma pack(push, 1)
union cn_slow_hash_state
{
    union hash_state hs;
    struct
    {
        uint8_t k[64];
        uint8_t init[INIT_SIZE_BYTE];
    };
};
#pragma pack(pop)

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define cpuid(info,x)    __cpuidex(info,x,0)
#else
void cpuid(int CPUInfo[4], int InfoType)
{
    __asm__ __volatile__
    (
        "cpuid":
        "=a" (CPUInfo[0]),
        "=b" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "a" (InfoType), "c" (0)
    );
}
#endif

STATIC INLINE void mul(const uint8_t *a, const uint8_t *b, uint8_t *res)
{
    uint64_t a0, b0;
    uint64_t hi, lo;

    a0 = U64(a)[0];
    b0 = U64(b)[0];
    lo = mul128(a0, b0, &hi);
    U64(res)[0] = hi;
    U64(res)[1] = lo;
}

STATIC INLINE void sum_half_blocks(uint8_t *a, const uint8_t *b)
{
    uint64_t a0, a1, b0, b1;
    a0 = U64(a)[0];
    a1 = U64(a)[1];
    b0 = U64(b)[0];
    b1 = U64(b)[1];
    a0 += b0;
    a1 += b1;
    U64(a)[0] = a0;
    U64(a)[1] = a1;
}

STATIC INLINE void swap_blocks(uint8_t *a, uint8_t *b)
{
    uint64_t t[2];
    U64(t)[0] = U64(a)[0];
    U64(t)[1] = U64(a)[1];
    U64(a)[0] = U64(b)[0];
    U64(a)[1] = U64(b)[1];
    U64(b)[0] = U64(t)[0];
    U64(b)[1] = U64(t)[1];
}

STATIC INLINE void xor_blocks(uint8_t *a, const uint8_t *b)
{
    U64(a)[0] ^= U64(b)[0];
    U64(a)[1] ^= U64(b)[1];
}

STATIC INLINE int check_aes_hw(void)
{
    int cpuid_results[4];
    static int supported = -1;

    if(supported >= 0)
        return supported;

    cpuid(cpuid_results,1);
    return supported = cpuid_results[2] & (1 << 25);
}

STATIC INLINE void aesni_pseudo_round(const uint8_t *in, uint8_t *out,
                                      const uint8_t *expandedKey)
{
    __m128i *k = R128(expandedKey);
    __m128i d;

    d = _mm_loadu_si128(R128(in));
    d = _mm_aesenc_si128(d, *R128(&k[0]));
    d = _mm_aesenc_si128(d, *R128(&k[1]));
    d = _mm_aesenc_si128(d, *R128(&k[2]));
    d = _mm_aesenc_si128(d, *R128(&k[3]));
    d = _mm_aesenc_si128(d, *R128(&k[4]));
    d = _mm_aesenc_si128(d, *R128(&k[5]));
    d = _mm_aesenc_si128(d, *R128(&k[6]));
    d = _mm_aesenc_si128(d, *R128(&k[7]));
    d = _mm_aesenc_si128(d, *R128(&k[8]));
    d = _mm_aesenc_si128(d, *R128(&k[9]));
    _mm_storeu_si128((R128(out)), d);
}

static inline void ExpandAESKey256_sub1(__m128i *tmp1, __m128i *tmp2)
{
	__m128i tmp4;
	*tmp2 = _mm_shuffle_epi32(*tmp2, 0xFF);
	tmp4 = _mm_slli_si128(*tmp1, 0x04);
	*tmp1 = _mm_xor_si128(*tmp1, tmp4);
	tmp4 = _mm_slli_si128(tmp4, 0x04);
	*tmp1 = _mm_xor_si128(*tmp1, tmp4);
	tmp4 = _mm_slli_si128(tmp4, 0x04);
	*tmp1 = _mm_xor_si128(*tmp1, tmp4);
	*tmp1 = _mm_xor_si128(*tmp1, *tmp2);
}

static inline void ExpandAESKey256_sub2(__m128i *tmp1, __m128i *tmp3)
{
	__m128i tmp2, tmp4;
	
	tmp4 = _mm_aeskeygenassist_si128(*tmp1, 0x00);
	tmp2 = _mm_shuffle_epi32(tmp4, 0xAA);
	tmp4 = _mm_slli_si128(*tmp3, 0x04);
	*tmp3 = _mm_xor_si128(*tmp3, tmp4);
	tmp4 = _mm_slli_si128(tmp4, 0x04);
	*tmp3 = _mm_xor_si128(*tmp3, tmp4);
	tmp4 = _mm_slli_si128(tmp4, 0x04);
	*tmp3 = _mm_xor_si128(*tmp3, tmp4);
	*tmp3 = _mm_xor_si128(*tmp3, tmp2);
}

// Special thanks to Intel for helping me
// with ExpandAESKey256() and its subroutines
static inline void ExpandAESKey256(char *keybuf)
{
	__m128i tmp1, tmp2, tmp3, *keys;
	
	keys = (__m128i *)keybuf;
	
	tmp1 = _mm_load_si128((__m128i *)keybuf);
	tmp3 = _mm_load_si128((__m128i *)(keybuf+0x10));
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x01);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[2] = tmp1;
	ExpandAESKey256_sub2(&tmp1, &tmp3);
	keys[3] = tmp3;
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x02);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[4] = tmp1;
	ExpandAESKey256_sub2(&tmp1, &tmp3);
	keys[5] = tmp3;
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x04);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[6] = tmp1;
	ExpandAESKey256_sub2(&tmp1, &tmp3);
	keys[7] = tmp3;
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x08);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[8] = tmp1;
	ExpandAESKey256_sub2(&tmp1, &tmp3);
	keys[9] = tmp3;
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x10);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[10] = tmp1;
	ExpandAESKey256_sub2(&tmp1, &tmp3);
	keys[11] = tmp3;
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x20);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[12] = tmp1;
	ExpandAESKey256_sub2(&tmp1, &tmp3);
	keys[13] = tmp3;
	
	tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x40);
	ExpandAESKey256_sub1(&tmp1, &tmp2);
	keys[14] = tmp1;
}

void cn_slow_hash(const void *data, size_t length, char *hash)
{
    uint8_t long_state[MEMORY];
    uint8_t text[INIT_SIZE_BYTE];
    uint64_t a[AES_BLOCK_SIZE >> 3];
    uint64_t b[AES_BLOCK_SIZE >> 3];
    uint8_t d[AES_BLOCK_SIZE];
    uint8_t aes_key[AES_KEY_SIZE];
    RDATA_ALIGN16 uint8_t expandedKey[256];

    union cn_slow_hash_state state;

    size_t i, j;
    uint8_t *p = NULL;
    oaes_ctx *aes_ctx;

    int useAes = check_aes_hw();
    static void (*const extra_hashes[4])(const void *, size_t, char *) =
    {
        hash_extra_blake, hash_extra_groestl, hash_extra_jh, hash_extra_skein
    };

    hash_process(&state.hs, data, length);
    
	memcpy(text, state.init, INIT_SIZE_BYTE);
	if(useAes)
	{
		memcpy(expandedKey, state.hs.b, AES_KEY_SIZE);
		ExpandAESKey256(expandedKey);
	}
	else
	{
		aes_ctx = (oaes_ctx *) oaes_alloc();
		oaes_key_import_data(aes_ctx, state.hs.b, AES_KEY_SIZE);
		memcpy(expandedKey, aes_ctx->key->exp_data, aes_ctx->key->exp_data_len);
	}
	
	__m128i *longoutput, *expkey, *xmminput;
	longoutput = (__m128i *)long_state;
	expkey = (__m128i *)expandedKey;
	xmminput = (__m128i *)text;
	
    if(useAes)
    {
		for (i = 0; __builtin_expect(i < MEMORY, 1); i += INIT_SIZE_BYTE)
		{
			for(j = 0; j < 10; j++)
			{
				xmminput[0] = _mm_aesenc_si128(xmminput[0], expkey[j]);
				xmminput[1] = _mm_aesenc_si128(xmminput[1], expkey[j]);
				xmminput[2] = _mm_aesenc_si128(xmminput[2], expkey[j]);
				xmminput[3] = _mm_aesenc_si128(xmminput[3], expkey[j]);
				xmminput[4] = _mm_aesenc_si128(xmminput[4], expkey[j]);
				xmminput[5] = _mm_aesenc_si128(xmminput[5], expkey[j]);
				xmminput[6] = _mm_aesenc_si128(xmminput[6], expkey[j]);
				xmminput[7] = _mm_aesenc_si128(xmminput[7], expkey[j]);
			}
			_mm_store_si128(&(longoutput[(i >> 4)]), xmminput[0]);
			_mm_store_si128(&(longoutput[(i >> 4) + 1]), xmminput[1]);
			_mm_store_si128(&(longoutput[(i >> 4) + 2]), xmminput[2]);
			_mm_store_si128(&(longoutput[(i >> 4) + 3]), xmminput[3]);
			_mm_store_si128(&(longoutput[(i >> 4) + 4]), xmminput[4]);
			_mm_store_si128(&(longoutput[(i >> 4) + 5]), xmminput[5]);
			_mm_store_si128(&(longoutput[(i >> 4) + 6]), xmminput[6]);
			_mm_store_si128(&(longoutput[(i >> 4) + 7]), xmminput[7]);
		}
    }
    else
    {
        for(i = 0; i < MEMORY / INIT_SIZE_BYTE; i++)
        {
            for(j = 0; j < INIT_SIZE_BLK; j++)
                aesb_pseudo_round(&text[AES_BLOCK_SIZE * j], &text[AES_BLOCK_SIZE * j], expandedKey);

            memcpy(&long_state[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE);
        }
    }
	
	for (i = 0; i < 2; i++) 
    {
	    a[i] = ((uint64_t *)state.k)[i] ^  ((uint64_t *)state.k)[i+4];
	    b[i] = ((uint64_t *)state.k)[i+2] ^  ((uint64_t *)state.k)[i+6];
    }
    
    if(useAes)
    {
		__m128i b_x = _mm_load_si128((__m128i *)b);
		
		for(i = 0; __builtin_expect(i < 0x80000, 1); i++)
		{	  
			__m128i c_x = _mm_load_si128((__m128i *)&long_state[a[0] & 0x1FFFF0]);
			__m128i a_x = _mm_load_si128((__m128i *)a);
			uint64_t c[2];
			c_x = _mm_aesenc_si128(c_x, a_x);

			_mm_store_si128((__m128i *)c, c_x);
			b_x = _mm_xor_si128(b_x, c_x);
			_mm_store_si128((__m128i *)&long_state[a[0] & 0x1FFFF0], b_x);

			uint64_t *nextblock = (uint64_t *)&long_state[c[0] & 0x1FFFF0];
			uint64_t b[2];
			b[0] = nextblock[0];
			b[1] = nextblock[1];

			{
			  uint64_t hi, lo;
			 // hi,lo = 64bit x 64bit multiply of c[0] and b[0]

			  __asm__("mulq %3\n\t"
				  : "=d" (hi),
				"=a" (lo)
				  : "%a" (c[0]),
				"rm" (b[0])
				  : "cc" );
			  
			  a[0] += hi;
			  a[1] += lo;
			}
			uint64_t *dst = &long_state[c[0] & 0x1FFFF0];
			dst[0] = a[0];
			dst[1] = a[1];

			a[0] ^= b[0];
			a[1] ^= b[1];
			b_x = c_x;
		}
	}
    else
    {
		for(i = 0; i < ITER / 2; i++)
		{
					#define TOTALBLOCKS (MEMORY / AES_BLOCK_SIZE)
					#define state_index(x) (((*((uint64_t *)x) >> 4) & (TOTALBLOCKS - 1)) << 4)

			// Iteration 1
			p = &long_state[state_index(a)];

			if(useAes)
				_mm_storeu_si128(R128(p), _mm_aesenc_si128(_mm_loadu_si128(R128(p)), _mm_loadu_si128(R128(a))));
			else
				aesb_single_round(p, p, a);

			xor_blocks(b, p);
			swap_blocks(b, p);
			swap_blocks(a, b);

			// Iteration 2
			p = &long_state[state_index(a)];

			mul(a, p, d);
			sum_half_blocks(b, d);
			swap_blocks(b, p);
			xor_blocks(b, p);
			swap_blocks(a, b);
		}
	}

    memcpy(text, state.init, INIT_SIZE_BYTE);
    
    if(useAes)
    {
		memcpy(expandedKey, &state.hs.b[32], AES_KEY_SIZE);
		ExpandAESKey256(expandedKey);
		
		for (i = 0; __builtin_expect(i < MEMORY, 1); i += INIT_SIZE_BYTE) 
		{	
			xmminput[0] = _mm_xor_si128(longoutput[(i >> 4)], xmminput[0]);
			xmminput[1] = _mm_xor_si128(longoutput[(i >> 4) + 1], xmminput[1]);
			xmminput[2] = _mm_xor_si128(longoutput[(i >> 4) + 2], xmminput[2]);
			xmminput[3] = _mm_xor_si128(longoutput[(i >> 4) + 3], xmminput[3]);
			xmminput[4] = _mm_xor_si128(longoutput[(i >> 4) + 4], xmminput[4]);
			xmminput[5] = _mm_xor_si128(longoutput[(i >> 4) + 5], xmminput[5]);
			xmminput[6] = _mm_xor_si128(longoutput[(i >> 4) + 6], xmminput[6]);
			xmminput[7] = _mm_xor_si128(longoutput[(i >> 4) + 7], xmminput[7]);
			
			for(j = 0; j < 10; j++)
			{
				xmminput[0] = _mm_aesenc_si128(xmminput[0], expkey[j]);
				xmminput[1] = _mm_aesenc_si128(xmminput[1], expkey[j]);
				xmminput[2] = _mm_aesenc_si128(xmminput[2], expkey[j]);
				xmminput[3] = _mm_aesenc_si128(xmminput[3], expkey[j]);
				xmminput[4] = _mm_aesenc_si128(xmminput[4], expkey[j]);
				xmminput[5] = _mm_aesenc_si128(xmminput[5], expkey[j]);
				xmminput[6] = _mm_aesenc_si128(xmminput[6], expkey[j]);
				xmminput[7] = _mm_aesenc_si128(xmminput[7], expkey[j]);
			}
		
		}
	}
	else
	{
		oaes_key_import_data(aes_ctx, &state.hs.b[32], AES_KEY_SIZE);
		memcpy(expandedKey, aes_ctx->key->exp_data, aes_ctx->key->exp_data_len);
		
		for(i = 0; i < MEMORY / INIT_SIZE_BYTE; i++)
        {
            for(j = 0; j < INIT_SIZE_BLK; j++)
            {
                xor_blocks(&text[j * AES_BLOCK_SIZE], &long_state[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
                aesb_pseudo_round(&text[AES_BLOCK_SIZE * j], &text[AES_BLOCK_SIZE * j], expandedKey);
            }
        }
        oaes_free((OAES_CTX **) &aes_ctx);
	}

    memcpy(state.init, text, INIT_SIZE_BYTE);
    hash_permutation(&state.hs);
    extra_hashes[state.hs.b[0] & 3](&state, 200, hash);
}
