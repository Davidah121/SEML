#include "SEML_256.h"
#include <cmath>
#include <iostream>

namespace SEML
{
    __m256 cmpgt(__m256 a, __m256 b)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        return _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_castps_si256(a), _mm256_castps_si256(b)));
    }
    __m256 cmplt(__m256 a, __m256 b)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        return _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_castps_si256(b), _mm256_castps_si256(a)));
    }
    __m256 cmpeq(__m256 a, __m256 b)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        return _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(a), _mm256_castps_si256(b)));
    }
    __m256d cmpgt(__m256d a, __m256d b)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        return _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_castpd_si256(a), _mm256_castpd_si256(b)));
    }
    __m256d cmplt(__m256d a, __m256d b)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        return _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_castpd_si256(b), _mm256_castpd_si256(a)));
    }
    __m256d cmpeq(__m256d a, __m256d b)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        return _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_castpd_si256(a), _mm256_castpd_si256(b)));
    }
    
    __m256 packDoublesIntoFloat(__m256d x1, __m256d x2)
    {
        __m128 outputLow = _mm256_cvtpd_ps(x1);
        __m128 outputHigh = _mm256_cvtpd_ps(x2);
        return _mm256_set_m128(outputHigh, outputLow); //just _mm256_insert
    }

    __m256i fastDoubleToInt64(__m256d x)
    {
        x = _mm256_add_pd(x, _mm256_set1_pd(0x0018000000000000));
        return _mm256_xor_si256(_mm256_castpd_si256(x), _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
    }
    __m256i fastDoubleToUInt64(__m256d x)
    {
        x = _mm256_add_pd(x, _mm256_set1_pd(0x0010000000000000));
        return _mm256_xor_si256(_mm256_castpd_si256(x), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
    }

    __m256d fastInt64ToDouble(__m256i x)
    {
        x = _mm256_add_epi64(x, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
        return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0018000000000000));
    }
    __m256d fastUInt64ToDouble(__m256i x)
    {
        x = _mm256_or_si256(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
        return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0010000000000000));
    }

    __m256d int64ToDouble(__m256i x)
    {
        __m256i xH = _mm256_srai_epi32(x, 16);
        xH = _mm256_blend_epi16(xH, _mm256_setzero_si256(), 0x33);
        xH = _mm256_add_epi64(xH, _mm256_castpd_si256(_mm256_set1_pd(442721857769029238784.)));              //  3*2^67
        __m256i xL = _mm256_blend_epi16(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)), 0x88);   //  2^52
        __m256d f = _mm256_sub_pd(_mm256_castsi256_pd(xH), _mm256_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
        return _mm256_add_pd(f, _mm256_castsi256_pd(xL));
    }
    __m256d uint64ToDouble(__m256i x)
    {
        __m256i xH = _mm256_srli_epi64(x, 32);
        xH = _mm256_or_si256(xH, _mm256_castpd_si256(_mm256_set1_pd(19342813113834066795298816.)));          //  2^84
        __m256i xL = _mm256_blend_epi16(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
        __m256d f = _mm256_sub_pd(_mm256_castsi256_pd(xH), _mm256_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
        return _mm256_add_pd(f, _mm256_castsi256_pd(xL));
    }

    //converts range to [-PI, PI]
    __m256 piRangeReduction(__m256 x)
    {
        const __m256 PI2DIV = _mm256_set1_ps(1.0/SEML_PI2);
        const __m256 PISSE = _mm256_set1_ps(SEML_PI);
        const __m256 PI2SSE = _mm256_set1_ps(SEML_PI2);
        __m256 xMinusPI = _mm256_sub_ps(x, PISSE);
        __m256 xMinus2PI = _mm256_sub_ps(x, PI2SSE);
        __m256 subValue = _mm256_mul_ps(PI2SSE, _mm256_floor_ps(_mm256_mul_ps(xMinusPI, PI2DIV)));

        return _mm256_sub_ps(xMinus2PI, subValue);
    }
    __m256d piRangeReduction(__m256d x)
    {
        const __m256d PI2DIV = _mm256_set1_pd(1.0/SEML_PI2);
        const __m256d PISSE = _mm256_set1_pd(SEML_PI);
        const __m256d PI2SSE = _mm256_set1_pd(SEML_PI2);
        __m256d xMinusPI = _mm256_sub_pd(x, PISSE);
        __m256d xMinus2PI = _mm256_sub_pd(x, PI2SSE);
        __m256d subValue = _mm256_mul_pd(PI2SSE, _mm256_floor_pd(_mm256_mul_pd(xMinusPI, PI2DIV)));

        return _mm256_sub_pd(xMinus2PI, subValue);
    }

    __m256 radToDeg(__m256 x)
    {
        const __m256 CONVERSION_MULT = _mm256_set1_ps(180.0/SEML_PI);
        return _mm256_mul_ps(x, CONVERSION_MULT);
    }
    __m256d radToDeg(__m256d x)
    {
        const __m256d CONVERSION_MULT = _mm256_set1_pd(180.0/SEML_PI);
        return _mm256_mul_pd(x, CONVERSION_MULT);
    }

    __m256 degToRad(__m256 x)
    {
        const __m256 CONVERSION_MULT = _mm256_set1_ps(SEML_PI/180.0);
        return _mm256_mul_ps(x, CONVERSION_MULT);
    }
    __m256d degToRad(__m256d x)
    {
        const __m256d CONVERSION_MULT = _mm256_set1_pd(SEML_PI/180.0);
        return _mm256_mul_pd(x, CONVERSION_MULT);
    }

    __m256 abs(__m256 x)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        const __m256i allButSignBit = _mm256_set1_epi32(0x7FFFFFFF);
        return _mm256_and_ps(x, _mm256_castsi256_ps(allButSignBit));
    }
    __m256d abs(__m256d x)
    {
        //NOTE: Only works with IEEE-754 double precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        const __m256i allButSignBit = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
        return _mm256_and_pd(x, _mm256_castsi256_pd(allButSignBit));
    }

    __m256 negate(__m256 x)
    {
        //NOTE: Only works with IEEE-754 single precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        const __m256i onlySignBit = _mm256_set1_epi32(0x80000000);
        return _mm256_xor_ps(x, _mm256_castsi256_ps(onlySignBit));
    }
    __m256d negate(__m256d x)
    {
        //NOTE: Only works with IEEE-754 double precision floats.
        //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
        const __m256i onlySignBit = _mm256_set1_epi64x(0x8000000000000000);
        return _mm256_xor_pd(x, _mm256_castsi256_pd(onlySignBit));
    }

    __m256 sign(__m256 x)
    {
        __m256 signMultBlend = cmplt(x, _mm256_set1_ps(0));
        __m256 signMult = _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(-1), signMultBlend);
        return signMult;
    }

    __m256d sign(__m256d x)
    {
        __m256d signMultBlend = cmplt(x, _mm256_set1_pd(0));
        __m256d signMult = _mm256_blendv_pd(_mm256_set1_pd(1), _mm256_set1_pd(-1), signMultBlend);
        return signMult;
    }
    
    __m256 reciprocal(__m256 a)
    {
        return _mm256_div_ps(_mm256_set1_ps(1), a);
    }
    __m256d reciprocal(__m256d a)
    {
        return _mm256_div_pd(_mm256_set1_pd(1), a);
    }
    __m256 fastReciprocal(__m256 a)
    {
        return _mm256_rcp_ps(a);
    }

    __m256 sqr(__m256 x)
    {
        return _mm256_mul_ps(x, x);
    }
    __m256d sqr(__m256d x)
    {
        return _mm256_mul_pd(x, x);
    }
    
	__m256 cube(__m256 a)
    {
        return _mm256_mul_ps(_mm256_mul_ps(a, a), a);
    }
	__m256d cube(__m256d a)
    {
        return _mm256_mul_pd(_mm256_mul_pd(a, a), a);
    }

    __m256 sqrt(__m256 x)
    {
        return _mm256_sqrt_ps(x);
    }
    __m256d sqrt(__m256d x)
    {
        return _mm256_sqrt_pd(x);
    }
    
    __m256 invSqrt(__m256 a)
    {
        return reciprocal(sqrt(a));
    }
    __m256d invSqrt(__m256d a)
    {
        return reciprocal(sqrt(a));
    }
    
    __m256 fastInvSqrt(__m256 a)
    {
        return _mm256_rsqrt_ps(a);
    }

    __m256d cosAround0(__m256d x)
    {
        const __m256d div1 = _mm256_set1_pd(1.0/2);
        const __m256d div2 = _mm256_set1_pd(1.0/24);
        const __m256d div3 = _mm256_set1_pd(1.0/720);
        const __m256d div4 = _mm256_set1_pd(1.0/40320);
        const __m256d div5 = _mm256_set1_pd(1.0/362880);
        // const __m256d div6 = _mm256_set1_pd(1.0/479001600);

        __m256d xSqr = _mm256_mul_pd(x, x);
        __m256d numerator = xSqr;
        __m256d sum = _mm256_set1_pd(1);

        sum = _mm256_sub_pd(sum, _mm256_mul_pd(numerator, div1));
        numerator = _mm256_mul_pd(xSqr, numerator);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(numerator, div2));
        numerator = _mm256_mul_pd(xSqr, numerator);
        sum = _mm256_sub_pd(sum, _mm256_mul_pd(numerator, div3));
        numerator = _mm256_mul_pd(xSqr, numerator);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(numerator, div4));
        numerator = _mm256_mul_pd(xSqr, numerator);
        sum = _mm256_sub_pd(sum, _mm256_mul_pd(numerator, div5));
        // numerator = _mm256_mul_pd(xSqr, numerator);
        // sum = _mm256_add_pd(sum, _mm256_mul_pd(numerator, div6));
        return sum;
    }

    std::pair<__m256, __m256> sincos(__m256 x)
    {
        const __m256 PIHalfHalf = _mm256_set1_ps(SEML_PI / 4);
        const __m256 PIHalf = _mm256_set1_ps(SEML_PI / 2);
        const __m256 PIHalfReci = _mm256_set1_ps(1.0 / (SEML_PI / 2));

        //reduce x into [-pi, pi]
        x = piRangeReduction(x);
        
        //record sign multipliers
        __m256 ltZero = cmplt(x, _mm256_set1_ps(0));
        __m256 gtPIHalf = cmpgt(abs(x), PIHalf);
        __m256 sinSign = _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(-1), ltZero);
        __m256 cosSign = _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(-1), gtPIHalf);

        //reduce x into [-pi/4, pi/4].
        __m256 floorAmount = _mm256_floor_ps( _mm256_mul_ps(_mm256_add_ps(x, PIHalfHalf), PIHalfReci));
        __m256i isEven = _mm256_and_si256(_mm256_cvtps_epi32(floorAmount), _mm256_set1_epi32(1)); //do something with this. If even, its cosine
        __m256 isCos = _mm256_castsi256_ps(_mm256_cmpeq_epi32(isEven, _mm256_set1_epi32(0)));
        
        __m256 xAdjusted = _mm256_sub_ps(x, _mm256_mul_ps(floorAmount, PIHalf));
        xAdjusted = abs(xAdjusted);

        //convert to double for higher accuracy
        __m256d lowHalf = _mm256_cvtps_pd(_mm256_extractf128_ps(xAdjusted, 0));
        __m256d highHalf = _mm256_cvtps_pd(_mm256_extractf128_ps(xAdjusted, 1));

        __m256d lowCosValue = cosAround0(lowHalf);
        __m256d highCosValue = cosAround0(highHalf);
        
        __m256d lowSinValue = sqrt(_mm256_sub_pd(_mm256_set1_pd(1), sqr(lowCosValue)));
        __m256d highSinValue = sqrt(_mm256_sub_pd(_mm256_set1_pd(1), sqr(highCosValue)));
        
        //convert back to floats
        __m256 cosValue = packDoublesIntoFloat(lowCosValue, highCosValue);
        __m256 sinValue = packDoublesIntoFloat(lowSinValue, highSinValue);

        //swap cosValue and sinValue depending on isCos.
        __m256 trueCosValue = _mm256_blendv_ps(sinValue, cosValue, isCos);
        __m256 trueSinValue = _mm256_blendv_ps(cosValue, sinValue, isCos);
        
        return {_mm256_mul_ps(sinSign, trueSinValue), _mm256_mul_ps(cosSign, trueCosValue)};
    }

    std::pair<__m256d, __m256d> sincos(__m256d x)
    {
        const __m256d PIHalfHalf = _mm256_set1_pd(SEML_PI / 4);
        const __m256d PIHalf = _mm256_set1_pd(SEML_PI / 2);
        const __m256d PIHalfReci = _mm256_set1_pd(1.0 / (SEML_PI / 2));

        //reduce x into [-pi, pi]
        x = piRangeReduction(x);
        
        //record sign multipliers
        __m256d ltZero = cmplt(x, _mm256_set1_pd(0));
        __m256d gtPIHalf = cmpgt(abs(x), PIHalf);
        __m256d sinSign = _mm256_blendv_pd(_mm256_set1_pd(1), _mm256_set1_pd(-1), ltZero);
        __m256d cosSign = _mm256_blendv_pd(_mm256_set1_pd(1), _mm256_set1_pd(-1), gtPIHalf);

        //reduce x into [-pi/4, pi/4].
        __m256d floorAmount = _mm256_floor_pd( _mm256_mul_pd(_mm256_add_pd(x, PIHalfHalf), PIHalfReci));
        __m256i isEven = _mm256_and_si256(fastDoubleToInt64(floorAmount), _mm256_set1_epi64x(1)); //do something with this. If even, its cosine
        __m256d isCos = _mm256_castsi256_pd(_mm256_cmpeq_epi64(isEven, _mm256_set1_epi64x(0)));
        
        __m256d xAdjusted = _mm256_sub_pd(x, _mm256_mul_pd(floorAmount, PIHalf));
        xAdjusted = abs(xAdjusted);

        //convert to double for higher accuracy
        __m256d cosValue = cosAround0(xAdjusted);
        __m256d sinValue = sqrt(_mm256_sub_pd(_mm256_set1_pd(1), sqr(cosValue)));

        //swap cosValue and sinValue depending on isCos.
        __m256d trueCosValue = _mm256_blendv_pd(sinValue, cosValue, isCos);
        __m256d trueSinValue = _mm256_blendv_pd(cosValue, sinValue, isCos);
        
        return {_mm256_mul_pd(sinSign, trueSinValue), _mm256_mul_pd(cosSign, trueCosValue)};
    }

    __m256 sin(__m256 x)
    {
        return sincos(x).first;
    }
    __m256d sin(__m256d x)
    {
        return sincos(x).first;
    }

    __m256 cos(__m256 x)
    {
        return sincos(x).second;
    }
    __m256d cos(__m256d x)
    {
        return sincos(x).second;
    }

    __m256 tan(__m256 x)
    {
        std::pair<__m256, __m256> values = sincos(x);
        return _mm256_div_ps(values.first, values.second);
    }
    __m256d tan(__m256d x)
    {
        std::pair<__m256d, __m256d> values = sincos(x);
        return _mm256_div_pd(values.first, values.second);
    }

    __m256 sec(__m256 x)
    {
        return _mm256_rcp_ps(sin(x));
    }
    __m256d sec(__m256d x)
    {
        return _mm256_div_pd(_mm256_set1_pd(1), sin(x));
    }

    __m256 csc(__m256 x)
    {
        return _mm256_rcp_ps(cos(x));
    }
    __m256d csc(__m256d x)
    {
        return _mm256_div_pd(_mm256_set1_pd(1), cos(x));
    }

    __m256 cot(__m256 x)
    {
        std::pair<__m256, __m256> values = sincos(x);
        return _mm256_div_ps(values.second, values.first);
    }
    __m256d cot(__m256d x)
    {
        std::pair<__m256d, __m256d> values = sincos(x);
        return _mm256_div_pd(values.second, values.first);
    }

    __m256d lnAround1(__m256d x)
    {
        //https://math.stackexchange.com/questions/977586/is-there-an-approximation-to-the-natural-log-function-at-large-values
        //Approximates the integral of x^t
        //Absolute error under the epsilon of single precision floating points (10^-6)
        //assume x is close to 1.
        const __m256d N1 = _mm256_set1_pd(90);
        const __m256d D1 = _mm256_set1_pd(7);
        const __m256d D2 = _mm256_set1_pd(32);
        const __m256d D3 = _mm256_set1_pd(12);
        const __m256d LN2 = _mm256_set1_pd(0.69314718056);

        //rule -> ln(x) = ln(x/2) + ln(2)
        x = _mm256_mul_pd(x, _mm256_set1_pd(0.5));

        //more instructions than a polynomial but far more accurate
        //2 ops
        __m256d numerator = _mm256_mul_pd(N1, _mm256_sub_pd(x, _mm256_set1_pd(1)));

        //2 ops of setup. potentially fast
        __m256d rootX = sqrt(x);
        __m256d fourthRoot = sqrt(rootX);
        
        //8 ops
        __m256d denominator = _mm256_mul_pd(D1, fourthRoot);
        denominator = _mm256_mul_pd(fourthRoot, _mm256_add_pd(D2, denominator));
        denominator = _mm256_mul_pd(fourthRoot, _mm256_add_pd(D3, denominator));
        denominator = _mm256_mul_pd(fourthRoot, _mm256_add_pd(D2, denominator));
        denominator = _mm256_add_pd(D1, denominator);

        __m256d result = _mm256_div_pd(numerator, denominator); //1 op - slow
        return _mm256_add_pd(result, LN2);
    }

    __m256 ln(__m256 x)
    {
        //note that at ln(x<=0) = -INF
        //constants found by fitting a polynomial to ln(x) on [1, 2]
        __m256i exponentExtraction = _mm256_set1_epi32(0x7F800000);
        __m256 LN2 = _mm256_set1_ps(0.69314718056f);
        //extract exponent from float
        //some type punning stuff
        __m256i exponent = _mm256_and_si256(_mm256_castps_si256(x), exponentExtraction);
        __m256 divisor = _mm256_castsi256_ps(exponent);
        __m256 exponentAdd = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(exponent, 23), _mm256_set1_epi32(127)));
        exponentAdd = _mm256_mul_ps(exponentAdd, LN2);

        //divide by exponent
        x = _mm256_div_ps(x, divisor);

        //approximate the fractional part between [1, 2]
        __m256d result1 = lnAround1(_mm256_cvtps_pd( _mm256_extractf128_ps(x, 0) ));
        __m256d result2 = lnAround1(_mm256_cvtps_pd( _mm256_extractf128_ps(x, 1) ));

        __m256 result = packDoublesIntoFloat(result1, result2);

        //add approximation and exponent
        result = _mm256_add_ps(result, exponentAdd);
        return result;
    }

    __m256d ln(__m256d x)
    {
        //note that at ln(x<=0) = -INF
        //constants found by fitting a polynomial to ln(x) on [1, 2]
        __m256i exponentExtraction = _mm256_set1_epi64x(0x7FF0000000000000);
        __m256d LN2 = _mm256_set1_pd(0.69314718056);
        //extract exponent from float
        //some type punning stuff
        __m256i exponent = _mm256_and_si256(_mm256_castpd_si256(x), exponentExtraction);
        __m256d divisor = _mm256_castsi256_pd(exponent);
        __m256d exponentAdd = int64ToDouble(_mm256_sub_epi64(_mm256_srli_epi64(exponent, 52), _mm256_set1_epi64x(1023)));
        exponentAdd = _mm256_mul_pd(exponentAdd, LN2);

        //divide by exponent
        x = _mm256_div_pd(x, divisor);

        //approximate the fractional part between [1, 2]
        __m256d result = lnAround1(x); //use better approximation

        //add approximation and exponent
        result = _mm256_add_pd(result, exponentAdd);
        return result;
    }

    __m256 log2(__m256 x)
    {
        const __m256 adjustment = _mm256_set1_ps(std::log(SEML_E)/std::log(2));
        return _mm256_mul_ps(ln(x), adjustment);
    }
    __m256d log2(__m256d x)
    {
        const __m256d adjustment = _mm256_set1_pd(std::log(SEML_E)/std::log(2));
        return _mm256_mul_pd(ln(x), adjustment);
    }

    __m256 log(__m256 x)
    {
        const __m256 adjustment = _mm256_set1_ps(std::log(SEML_E));
        return _mm256_mul_ps(ln(x), adjustment);
    }
    __m256d log(__m256d x)
    {
        const __m256d adjustment = _mm256_set1_pd(std::log(SEML_E));
        return _mm256_mul_pd(ln(x), adjustment);
    }

    __m256 log(__m256 x, float base)
    {
        __m256 adjustment = _mm256_set1_ps(std::log(SEML_E) / std::log(base));
        return _mm256_mul_ps(ln(x), adjustment);
    }
    __m256d log(__m256d x, double base)
    {
        __m256d adjustment = _mm256_set1_pd(std::log(SEML_E) / std::log(base));
        return _mm256_mul_pd(ln(x), adjustment);
    }

    __m256 log(__m256 x, __m256 base)
    {
        __m256 adjustment = _mm256_set1_ps(std::log(SEML_E));
        adjustment = _mm256_div_ps(adjustment, log(base));
        return _mm256_mul_ps(ln(x), adjustment);
    }
    __m256d log(__m256d x, __m256d base)
    {
        __m256d adjustment = _mm256_set1_pd(std::log(SEML_E));
        adjustment = _mm256_div_pd(adjustment, log(base));
        return _mm256_mul_pd(ln(x), adjustment);
    }

    __m256d expAround0(__m256d x)
    {
        //sigh... use double precision and a taylor series of like 11 terms
        __m256d result = _mm256_set1_pd(1);
        __m256d numerator = x;
        __m256d denominator = _mm256_set1_pd(1);
        
        for(int i=1; i<=11; i++)
        {
            __m256d addV = _mm256_div_pd(numerator, denominator);
            
            result = _mm256_add_pd(addV, result);
            numerator = _mm256_mul_pd(numerator, x);
            denominator = _mm256_mul_pd(denominator, _mm256_set1_pd(i+1));
        }

        return result;
    }

    __m256d exp(__m256d x)
    {
        const __m256d log2E = _mm256_set1_pd(1.44269504089); //log base 2 (e)
        const __m256d invLog2E = _mm256_set1_pd(1.0 / 1.44269504089); //log base 2 (e)

        //so extract the whole number part of the exponent and the fractional part
        __m256d temp = _mm256_mul_pd(x, log2E);
        __m256d exponent = _mm256_floor_pd(temp); //the exponent part
        __m256i exponentAsInt = _mm256_cvtepu32_epi64(_mm256_cvtpd_epi32(exponent)); //dumb but required

        __m256d fraction = _mm256_sub_pd(temp, exponent); //fractional part
        fraction = _mm256_mul_pd(fraction, invLog2E);

        //approximate the fractional part
        __m256d result = expAround0(fraction);
        
        //multiply them together
        __m256i expTemp = _mm256_slli_epi64(exponentAsInt, 52); //preparation to convert back to double
        
        //cheating
        //multiplying by power of 2 is equivalent to adding to the exponent of the float
        //add to the exponent then cast back. Potential problem for very very high values
        result = _mm256_castsi256_pd(_mm256_add_epi64(expTemp, _mm256_castpd_si256(result)));
        return result;
    }

    __m256 exp(__m256 x)
    {
        //lazy. could potentially be faster if both are intertwined
        __m256d lowValues = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 0));
        __m256d highValues = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
        __m256d result1 = exp(lowValues);
        __m256d result2 = exp(highValues);
        return packDoublesIntoFloat(result1, result2);
    }

    __m256 pow(__m256 x, float power)
    {
        //already inaccurate but normal pow(x, y) is also inaccurate
        //e^(power*ln(x))
        __m256 blendMask = cmpeq(x, _mm256_set1_ps(0));
        __m256 result = exp( _mm256_mul_ps(_mm256_set1_ps(power), ln(x)));
        result = _mm256_blendv_ps(result, _mm256_set1_ps(0), blendMask);
        return result;
    }
    __m256d pow(__m256d x, double power)
    {
        //e^(power*ln(x))
        __m256d blendMask = cmpeq(x, _mm256_set1_pd(0));
        __m256d result = exp( _mm256_mul_pd(_mm256_set1_pd(power), ln(x)));
        result = _mm256_blendv_pd(result, _mm256_set1_pd(0), blendMask);
        return result;
    }

    __m256 pow(__m256 x, __m256 power)
    {
        //e^(power*ln(x))
        __m256 blendMask = cmpeq(x, _mm256_set1_ps(0));
        __m256 result = exp( _mm256_mul_ps(power, ln(x)));
        result = _mm256_blendv_ps(result, _mm256_set1_ps(0), blendMask);
        return result;
    }
    __m256d pow(__m256d x, __m256d power)
    {
        //e^(power*ln(x))
        __m256d blendMask = cmpeq(x, _mm256_set1_pd(0));
        __m256d result = exp( _mm256_mul_pd(power, ln(x)));
        result = _mm256_blendv_pd(result, _mm256_set1_pd(0), blendMask);
        return result;
    }

    __m256 arcsin(__m256 x)
    {
        //arctan of adjusted input: x / sqrt(1 - x^2)
        __m256 newInput = sqrt( _mm256_sub_ps(_mm256_set1_ps(1), sqr(x)) );
        newInput = _mm256_div_ps(x, newInput);
        __m256 result = arctan(newInput);
        //special case. if x == 1, result is pi/2. if x==-1, result is -pi/2
        __m256 blendValue = cmpeq(abs(x), _mm256_set1_ps(1));
        __m256 possibleOutput = _mm256_mul_ps(_mm256_set1_ps(SEML_PI/2), sign(x));

        return _mm256_blendv_ps(result, possibleOutput, blendValue);
    }
    __m256d arcsin(__m256d x)
    {
        //arctan of adjusted input: x / sqrt(1 - x^2)
        __m256d newInput = sqrt( _mm256_sub_pd(_mm256_set1_pd(1), sqr(x)) );
        newInput = _mm256_div_pd(x, newInput);
        __m256d result = arctan(newInput);
        //special case. if x == 1, result is pi/2. if x==-1, result is -pi/2
        __m256d blendValue = cmpeq(abs(x), _mm256_set1_pd(1));
        __m256d possibleOutput = _mm256_mul_pd(_mm256_set1_pd(SEML_PI/2), sign(x));

        return _mm256_blendv_pd(result, possibleOutput, blendValue);
    }

    __m256 arccos(__m256 x)
    {
        const __m256 PIDIV2 = _mm256_set1_ps(SEML_PI/2);
        return _mm256_sub_ps(PIDIV2, arcsin(x));
    }
    __m256d arccos(__m256d x)
    {
        const __m256d PIDIV2 = _mm256_set1_pd(SEML_PI/2);
        return _mm256_sub_pd(PIDIV2, arcsin(x));
    }

    __m256d arctanApproxHigherThan1(__m256d x)
    {
        //https://mae.ufl.edu/~uhk/ARCTAN-APPROX-PAPER.pdf
        //using n=8 instead of n=4. solved manually for n=8
        const __m256d N1 = _mm256_set1_pd(307835);
        const __m256d N2 = _mm256_set1_pd(4813380);
        const __m256d N3 = _mm256_set1_pd(19801782);
        const __m256d N4 = _mm256_set1_pd(29609580);
        const __m256d N5 = _mm256_set1_pd(14549535);

        const __m256d D1 = _mm256_set1_pd(19845);
        const __m256d D2 = _mm256_set1_pd(1091475);
        const __m256d D3 = _mm256_set1_pd(9459450);
        const __m256d D4 = _mm256_set1_pd(28378350);
        const __m256d D5 = _mm256_set1_pd(34459425);
        const __m256d D6 = _mm256_set1_pd(14549535);

        __m256d xSqr = sqr(x);
        __m256d numerator = _mm256_add_pd(N4, _mm256_mul_pd(xSqr, N5));
        numerator = _mm256_add_pd(N3, _mm256_mul_pd(xSqr, numerator));
        numerator = _mm256_add_pd(N2, _mm256_mul_pd(xSqr, numerator));
        numerator = _mm256_add_pd(N1, _mm256_mul_pd(xSqr, numerator));

        __m256d denominator = _mm256_add_pd(D5, _mm256_mul_pd(xSqr, D6));
        denominator = _mm256_add_pd(D4, _mm256_mul_pd(xSqr, denominator));
        denominator = _mm256_add_pd(D3, _mm256_mul_pd(xSqr, denominator));
        denominator = _mm256_add_pd(D2, _mm256_mul_pd(xSqr, denominator));
        denominator = _mm256_add_pd(D1, _mm256_mul_pd(xSqr, denominator));
        return _mm256_div_pd(numerator, denominator);
    }

    __m256 arctan(__m256 x)
    {
        __m256 signMult = sign(x);
        x = abs(x);
        __m256 isZero = cmpeq(x, _mm256_set1_ps(0)); //if 0, return 0
        __m256 ltBlend = cmplt(x, _mm256_set1_ps(1)); //if less than 1, use 1/x
        __m256 adjustment = _mm256_blendv_ps(_mm256_set1_ps(SEML_PI/2), _mm256_set1_ps(0), ltBlend); //at the end, pi/2 - approx or 0 - approx
        x = _mm256_blendv_ps(x, _mm256_div_ps(_mm256_set1_ps(1), x), ltBlend); //sigh... rcp_ps is too inaccurate to use here

        __m256d tempX = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 0));
        __m256d tempX2 = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

        __m256d result1 = arctanApproxHigherThan1(tempX);
        __m256d result2 = arctanApproxHigherThan1(tempX2);
        __m256 finalResult = _mm256_mul_ps(x, packDoublesIntoFloat(result1, result2));

        //multiply sign into |adjustment-result|
        finalResult = _mm256_mul_ps(signMult, abs(_mm256_sub_ps(adjustment, finalResult)));
        finalResult = _mm256_blendv_ps(finalResult, _mm256_set1_ps(0), isZero); //x=0 must be accounted for
        return finalResult;
    }
    __m256d arctan(__m256d x)
    {
        __m256d signMult = sign(x);
        x = abs(x);
        __m256d isZero = cmpeq(x, _mm256_set1_pd(0)); //if 0, return 0
        __m256d ltBlend = cmplt(x, _mm256_set1_pd(1)); //if less than 1, use 1/x
        __m256d adjustment = _mm256_blendv_pd(_mm256_set1_pd(SEML_PI/2), _mm256_set1_pd(0), ltBlend); //at the end, pi/2 - approx or 0 - approx
        x = _mm256_blendv_pd(x, _mm256_div_pd(_mm256_set1_pd(1), x), ltBlend);

        __m256d finalResult = _mm256_mul_pd(x, arctanApproxHigherThan1(x));
        
        //multiply sign into |adjustment-result|
        finalResult = _mm256_mul_pd(signMult, abs(_mm256_sub_pd(adjustment, finalResult)));
        finalResult = _mm256_blendv_pd(finalResult, _mm256_set1_pd(0), isZero); //x=0 must be accounted for
        return finalResult;
    }

    __m256 arccsc(__m256 x)
    {
        return arcsin(_mm256_rcp_ps(x));
    }
    __m256d arccsc(__m256d x)
    {
        return arcsin(_mm256_div_pd(_mm256_set1_pd(1), x));
    }

    __m256 arcsec(__m256 x)
    {
        return arccos(_mm256_rcp_ps(x));
    }
    __m256d arcsec(__m256d x)
    {
        return arccos(_mm256_div_pd(_mm256_set1_pd(1), x));
    }

    __m256 arccot(__m256 x)
    {
        const __m256 PIDIV2 = _mm256_set1_ps(SEML_PI/2);
        return _mm256_sub_ps(PIDIV2, arctan(x));
    }
    __m256d arccot(__m256d x)
    {
        const __m256d PIDIV2 = _mm256_set1_pd(SEML_PI/2);
        return _mm256_sub_pd(PIDIV2, arctan(x));
    }

    __m256 sinh(__m256 x)
    {
        __m256 eX = exp(x);
        __m256 negativeEX = reciprocal(eX);
        __m256 numerator = _mm256_sub_ps(eX, negativeEX);
        return _mm256_mul_ps(numerator, _mm256_set1_ps(0.5));
    }
    __m256d sinh(__m256d x)
    {
        __m256d eX = exp(x);
        __m256d negativeEX = reciprocal(eX);
        __m256d numerator = _mm256_sub_pd(eX, negativeEX);
        return _mm256_mul_pd(numerator, _mm256_set1_pd(0.5));
    }

    __m256 cosh(__m256 x)
    {
        __m256 eX = exp(x);
        __m256 negativeEX = reciprocal(eX);
        __m256 numerator = _mm256_add_ps(eX, negativeEX);
        return _mm256_mul_ps(numerator, _mm256_set1_ps(0.5));
    }
    __m256d cosh(__m256d x)
    {
        __m256d eX = exp(x);
        __m256d negativeEX = reciprocal(eX);
        __m256d numerator = _mm256_add_pd(eX, negativeEX);
        return _mm256_mul_pd(numerator, _mm256_set1_pd(0.5));
    }

    __m256 tanh(__m256 x)
    {
        __m256 expValue = exp(_mm256_mul_ps(x, _mm256_set1_ps(2)));
        __m256 numerator = _mm256_sub_ps(expValue, _mm256_set1_ps(1));
        __m256 denominator = _mm256_add_ps(expValue, _mm256_set1_ps(1));
        return _mm256_div_ps(numerator, denominator);
    }
    __m256d tanh(__m256d x)
    {
        __m256d expValue = exp(_mm256_mul_pd(x, _mm256_set1_pd(2)));
        __m256d numerator = _mm256_sub_pd(expValue, _mm256_set1_pd(1));
        __m256d denominator = _mm256_add_pd(expValue, _mm256_set1_pd(1));
        return _mm256_div_pd(numerator, denominator);
    }

    __m256 sech(__m256 x)
    {
        __m256 eX = exp(x);
        __m256 negativeEX = reciprocal(eX);
        __m256 denominator = _mm256_add_ps(eX, negativeEX);
        return _mm256_div_ps(_mm256_set1_ps(0.5), denominator);
    }
    __m256d sech(__m256d x)
    {
        __m256d eX = exp(x);
        __m256d negativeEX = reciprocal(eX);
        __m256d denominator = _mm256_add_pd(eX, negativeEX);
        return _mm256_div_pd(_mm256_set1_pd(0.5), denominator);
    }

    __m256 csch(__m256 x)
    {
        __m256 eX = exp(x);
        __m256 negativeEX = reciprocal(eX);
        __m256 denominator = _mm256_sub_ps(eX, negativeEX);
        return _mm256_div_ps(_mm256_set1_ps(0.5), denominator);
    }
    __m256d csch(__m256d x)
    {
        __m256d eX = exp(x);
        __m256d negativeEX = reciprocal(eX);
        __m256d denominator = _mm256_sub_pd(eX, negativeEX);
        return _mm256_div_pd(_mm256_set1_pd(0.5), denominator);
    }

    __m256 coth(__m256 x)
    {
        __m256 expValue = exp(_mm256_mul_ps(x, _mm256_set1_ps(2)));
        __m256 numerator = _mm256_add_ps(expValue, _mm256_set1_ps(1));
        __m256 denominator = _mm256_sub_ps(expValue, _mm256_set1_ps(1));
        return _mm256_div_ps(numerator, denominator);
    }
    __m256d coth(__m256d x)
    {
        __m256d expValue = exp(_mm256_mul_pd(x, _mm256_set1_pd(2)));
        __m256d numerator = _mm256_add_pd(expValue, _mm256_set1_pd(1));
        __m256d denominator = _mm256_sub_pd(expValue, _mm256_set1_pd(1));
        return _mm256_div_pd(numerator, denominator);
    }

    __m256 arcsinh(__m256 x)
    {
        __m256 value = sqrt(_mm256_add_ps(x, _mm256_add_ps(sqr(x), _mm256_set1_ps(1))));
        return ln(value);
    }
    __m256d arcsinh(__m256d x)
    {
        __m256d value = sqrt(_mm256_add_pd(x, _mm256_add_pd(sqr(x), _mm256_set1_pd(1))));
        return ln(value);
    }

    __m256 arccosh(__m256 x)
    {
        __m256 value = sqrt(_mm256_add_ps(x, _mm256_sub_ps(sqr(x), _mm256_set1_ps(1))));
        return ln(value);
    }
    __m256d arccosh(__m256d x)
    {
        __m256d value = sqrt(_mm256_add_pd(x, _mm256_sub_pd(sqr(x), _mm256_set1_pd(1))));
        return ln(value);
    }

    __m256 arctanh(__m256 x)
    {
        __m256 numerator = _mm256_add_ps(_mm256_set1_ps(1), x);
        __m256 denominator = _mm256_sub_ps(_mm256_set1_ps(1), x);
        return _mm256_mul_ps(_mm256_set1_ps(0.5), ln(_mm256_div_ps(numerator, denominator)));
    }
    __m256d arctanh(__m256d x)
    {
        __m256d numerator = _mm256_add_pd(_mm256_set1_pd(1), x);
        __m256d denominator = _mm256_sub_pd(_mm256_set1_pd(1), x);
        return _mm256_mul_pd(_mm256_set1_pd(0.5), ln(_mm256_div_pd(numerator, denominator)));
    }

    __m256 arccsch(__m256 x)
    {
        return arcsinh(_mm256_rcp_ps(x));
    }
    __m256d arccsch(__m256d x)
    {
        return arcsinh(_mm256_div_pd(_mm256_set1_pd(1), x));
    }

    __m256 arcsech(__m256 x)
    {
        return arccosh(_mm256_rcp_ps(x));
    }
    __m256d arcsech(__m256d x)
    {
        return arccosh(_mm256_div_pd(_mm256_set1_pd(1), x));
    }

    __m256 arccoth(__m256 x)
    {
        __m256 numerator = _mm256_add_ps(x, _mm256_set1_ps(1));
        __m256 denominator = _mm256_sub_ps(x, _mm256_set1_ps(1));
        return _mm256_mul_ps(_mm256_set1_ps(0.5), ln(_mm256_div_ps(numerator, denominator)));
    }
    __m256d arccoth(__m256d x)
    {
        __m256d numerator = _mm256_add_pd(x, _mm256_set1_pd(1));
        __m256d denominator = _mm256_sub_pd(x, _mm256_set1_pd(1));
        return _mm256_mul_pd(_mm256_set1_pd(0.5), ln(_mm256_div_pd(numerator, denominator)));
    }
}