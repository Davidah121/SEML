#include "SEML.h"
#include <cmath>
#include <iostream>

//converts range to [-PI, PI]
__m128 piRangeReduction(__m128 x)
{
    const __m128 PI2DIV = _mm_set1_ps(1.0/SEML_PI2);
    const __m128 PISSE = _mm_set1_ps(SEML_PI);
    const __m128 PI2SSE = _mm_set1_ps(SEML_PI2);
    __m128 xMinusPI = _mm_sub_ps(x, PISSE);
    __m128 xMinus2PI = _mm_sub_ps(x, PI2SSE);
    __m128 subValue = _mm_mul_ps(PI2SSE, _mm_floor_ps(_mm_mul_ps(xMinusPI, PI2DIV)));

    return _mm_sub_ps(xMinus2PI, subValue);
}

__m128 radToDeg(__m128 x)
{
    const __m128 CONVERSION_MULT = _mm_set1_ps(180.0/SEML_PI);
    return _mm_mul_ps(x, CONVERSION_MULT);
}

__m128 degToRad(__m128 x)
{
    const __m128 CONVERSION_MULT = _mm_set1_ps(SEML_PI/180.0);
    return _mm_mul_ps(x, CONVERSION_MULT);
}

__m128 abs(__m128 x)
{
    //NOTE: Only works with IEEE-754 single precision floats.
    //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
    const __m128i allButSignBit = _mm_set1_epi32(0x7FFFFFFF);
    return _mm_and_ps(x, _mm_castsi128_ps(allButSignBit));
}

__m128 negate(__m128 x)
{
    //NOTE: Only works with IEEE-754 single precision floats.
    //Should be valid on all x86 processors especially if it uses SSE / AVX / etc.
    const __m128i onlySignBit = _mm_set1_epi32(0x80000000);
    return _mm_xor_ps(x, _mm_castsi128_ps(onlySignBit));
}

__m128 sqr(__m128 x)
{
    return _mm_mul_ps(x, x);
}

__m128 sqrt(__m128 x)
{
    return _mm_sqrt_ps(x);
}

__m128 sin(__m128 x)
{
    //NOTE: Pick better approximation for sin potentially

    //solve quartic poly for range [0, pi/2]
    //expand [0, pi/2] -> [0, pi] through duplication
    //expand [0, pi] -> [-pi, pi] through sign preservation
    const __m128 PISSE = _mm_set1_ps(SEML_PI);
    const __m128 PIHALFDIVSSE = _mm_set1_ps(1.0 / (SEML_PI/2.0));
    const __m128 A = _mm_set1_ps(0.028713);
    const __m128 B = _mm_set1_ps(-0.203586);
    const __m128 C = _mm_set1_ps(0.019954);
    const __m128 D = _mm_set1_ps(0.996317);
    
    //now x in [-pi, pi]
    __m128 rangeReduced = piRangeReduction(x);
    
    __m128 gtZero = _mm_cmplt_ps(rangeReduced, _mm_set1_ps(0));
    __m128 signMultipliers = _mm_blendv_ps(_mm_set1_ps(1), _mm_set1_ps(-1), gtZero);

    //now x in [0, pi]
    __m128 absX = abs(rangeReduced); //assuming this works

    //now x in [0, pi/2]
    __m128 fullyReduced = _mm_mul_ps(PISSE, _mm_floor_ps( _mm_mul_ps(absX, PIHALFDIVSSE)));
    fullyReduced = abs(_mm_sub_ps(fullyReduced, absX));

    //solve quartic polynomial. Note that E = 0 so its omitted
    //Instead of Ax^4 + Bx^3 + Cx^2 + Dx
    //x(D + x(C + x(B + x(A))))

    __m128 result = _mm_mul_ps(fullyReduced, A);
    result = _mm_mul_ps(fullyReduced, _mm_add_ps(B, result));
    result = _mm_mul_ps(fullyReduced, _mm_add_ps(C, result));
    result = _mm_mul_ps(fullyReduced, _mm_add_ps(D, result));
    
    //due to symmetry, we have solved [0, pi]
    //now negate if necessary to get [-pi, pi]
    return _mm_mul_ps(result, signMultipliers);
}

__m128 cos(__m128 x)
{
    return sin(_mm_sub_ps(_mm_set1_ps(SEML_PI/2), x));
}

__m128 tan(__m128 x)
{
    __m128 sinValue = sin(x);
    __m128 cosValue = sqrt(_mm_sub_ps(_mm_set1_ps(1), sqr(sinValue)));

    return _mm_div_ps(sinValue, cosValue);
}

__m128 sec(__m128 x)
{
    return _mm_rcp_ps(sin(x));
}

__m128 csc(__m128 x)
{
    return _mm_rcp_ps(cos(x));
}

__m128 cot(__m128 x)
{
    __m128 sinValue = sin(x);
    __m128 cosValue = sqrt(_mm_sub_ps(_mm_set1_ps(1), sqr(sinValue)));

    return _mm_div_ps(cosValue, sinValue);
}

__m128 lnAround1(__m128 x)
{
    //https://math.stackexchange.com/questions/977586/is-there-an-approximation-to-the-natural-log-function-at-large-values
    //Approximates the integral of x^t
    //Absolute error under the epsilon of single precision floating points (10^-6)
    //assume x is close to 1.
    const __m128 N1 = _mm_set1_ps(90);
    const __m128 D1 = _mm_set1_ps(7);
    const __m128 D2 = _mm_set1_ps(32);
    const __m128 D3 = _mm_set1_ps(12);

    //more instructions than a polynomial but far more accurate
    //2 ops
    __m128 numerator = _mm_mul_ps(N1, _mm_sub_ps(x, _mm_set1_ps(1)));

    //2 ops of setup. potentially fast
    __m128 rootX = sqrt(x);
    __m128 fourthRoot = sqrt(rootX);
    
    //8 ops
    __m128 denominator = _mm_mul_ps(D1, fourthRoot);
    denominator = _mm_mul_ps(fourthRoot, _mm_add_ps(D2, denominator));
    denominator = _mm_mul_ps(fourthRoot, _mm_add_ps(D3, denominator));
    denominator = _mm_mul_ps(fourthRoot, _mm_add_ps(D2, denominator));
    denominator = _mm_add_ps(D1, denominator);

    return _mm_div_ps(numerator, denominator); //1 op - slow
}

__m128 ln(__m128 x)
{
    //note that at ln(x<=0) = -INF
    //constants found by fitting a polynomial to ln(x) on [1, 2]
    __m128i exponentExtraction = _mm_set1_epi32(0x7F800000);
    __m128 LN2 = _mm_set1_ps(0.69314718056f);
    //extract exponent from float
    //some type punning stuff
    __m128i exponent = _mm_and_si128(_mm_castps_si128(x), exponentExtraction);
    __m128 divisor = _mm_castsi128_ps(exponent);
    __m128 exponentAdd = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(exponent, 23), _mm_set1_epi32(127)));
    exponentAdd = _mm_mul_ps(exponentAdd, LN2);

    //divide by exponent
    x = _mm_div_ps(x, divisor);

    //approximate the fractional part between [1, 2]
    __m128 result = lnAround1(x);

    //add approximation and exponent
    result = _mm_add_ps(result, exponentAdd);
    return result;
}

__m128 log2(__m128 x)
{
    const __m128 adjustment = _mm_set1_ps(std::log(SEML_E)/std::log(2));
    return _mm_mul_ps(ln(x), adjustment);
}

__m128 log(__m128 x)
{
    const __m128 adjustment = _mm_set1_ps(std::log(SEML_E));
    return _mm_mul_ps(ln(x), adjustment);
}

__m128 log(__m128 x, float base)
{
    __m128 adjustment = _mm_set1_ps(std::log(SEML_E) / std::log(base));
    return _mm_mul_ps(ln(x), adjustment);
}

__m128 log(__m128 x, __m128 base)
{
    __m128 adjustment = _mm_set1_ps(std::log(SEML_E));
    adjustment = _mm_div_ps(adjustment, log(base));
    return _mm_mul_ps(ln(x), adjustment);
}

__m128 exp2Around0(__m128 x)
{
    //5th degree polynomial instead of pade for better performance potentially
    //Not minimax to ensure that f(0) == 1. High accuracy between [0, 1]
    //constants for a polynomial fitting 2^x from [0, 0.5]
    const __m128 A = _mm_set1_ps(0.00158722f);
    const __m128 B = _mm_set1_ps(0.00946996f);
    const __m128 C = _mm_set1_ps(0.05554388f);
    const __m128 D = _mm_set1_ps(0.24022163f);
    const __m128 E = _mm_set1_ps(0.6931474f);
    const __m128 F = _mm_set1_ps(1.0f);

    //trick: 2^(Y1+Y2) = 2^Y1 * 2^Y2
    //let Y1 = Y2 = x/2
    //solve 2^(x/2)
    __m128 nX = _mm_mul_ps(x, _mm_set1_ps(0.5));
    __m128 result = _mm_mul_ps(nX, A);
    result = _mm_mul_ps(nX, _mm_add_ps(B, result));
    result = _mm_mul_ps(nX, _mm_add_ps(C, result));
    result = _mm_mul_ps(nX, _mm_add_ps(D, result));
    result = _mm_mul_ps(nX, _mm_add_ps(E, result));
    result = _mm_add_ps(F, result);

    //return result^2
    return sqr(result);
}

__m128 exp(__m128 x)
{
    //https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
    const __m128 log2E = _mm_set1_ps(1.44269504089f); //log base 2 (e)
    //exp(x) = 2^i * 2^f
    //so extract the whole number part of the exponent and the fractional part
    __m128 temp = _mm_mul_ps(x, log2E);

    __m128 wholeNumberPart = _mm_floor_ps(temp); //the exponent part
    __m128i exponentOnly = _mm_cvtps_epi32(wholeNumberPart);

    __m128 fraction = _mm_sub_ps(temp, wholeNumberPart); //fractional part

    //approximate the fractional part
    __m128 result = exp2Around0(fraction);
    
    //multiply them together
    __m128i expTemp = _mm_slli_epi32(exponentOnly, 23); //preparation to convert back to float
    
    //cheating
    //multiplying by power of 2 is equivalent to adding to the exponent of the float
    //add to the exponent then cast back. Potential problem for very very high values
    result = _mm_castsi128_ps(_mm_add_epi32(expTemp, _mm_castps_si128(result)));
    return result;
}

__m128 pow(__m128 x, float power)
{
    //e^(power*ln(x))
    __m128 blendMask = _mm_cmpeq_ps(x, _mm_set1_ps(0));
    __m128 result = exp( _mm_mul_ps(_mm_set1_ps(power), ln(x)));
    result = _mm_blendv_ps(result, _mm_set1_ps(0), blendMask);
    return result;
}

__m128 pow(__m128 x, __m128 power)
{
    //e^(power*ln(x))
    __m128 blendMask = _mm_cmpeq_ps(x, _mm_set1_ps(0));
    __m128 result = exp( _mm_mul_ps(power, ln(x)));
    result = _mm_blendv_ps(result, _mm_set1_ps(0), blendMask);
    return result;
}

__m128 arcsin(__m128 x)
{
    //arctan approach but swap at different point along with adjusting input
    const __m128 A = _mm_set1_ps(-0.00749305860992f);
    const __m128 B = _mm_set1_ps(0.03252232640125f);
    const __m128 C = _mm_set1_ps(-0.08467922817644f);
    const __m128 D = _mm_set1_ps(0.33288950512027f);
    const __m128 E = _mm_set1_ps(1.0f);
    const __m128 PIDIV2 = _mm_set1_ps(SEML_PI/2);

    //extract sign
    __m128 signMult = _mm_blendv_ps(_mm_set1_ps(1), _mm_set1_ps(-1), _mm_cmplt_ps(x, _mm_set1_ps(0))); //multiply by the result
    x = abs(x);

    //rewrite input and determine if it is 1.
    __m128 blendV = _mm_cmpgt_ps(x, _mm_set1_ps(0.707106781f)); // 1/sqrt(2)
    __m128 forcePIDIV2 = _mm_cmpeq_ps(x, _mm_set1_ps(1)); //may need to set to PI/2 at the end
    x = _mm_div_ps(x, sqrt(_mm_sub_ps(_mm_set1_ps(1), sqr(x)))); //x / sqrt(1 - x^2)

    x = _mm_blendv_ps(x, _mm_rcp_ps(x), blendV);

    __m128 xSqr = sqr(x);
    __m128 divisor = _mm_mul_ps(A, xSqr);
    divisor = _mm_mul_ps(xSqr, _mm_add_ps(B, divisor));
    divisor = _mm_mul_ps(xSqr, _mm_add_ps(C, divisor));
    divisor = _mm_mul_ps(xSqr, _mm_add_ps(D, divisor));
    divisor = _mm_add_ps(E, divisor);
    
    //Correct result for [-1, 1]
    __m128 result = _mm_div_ps(x, divisor);

    //otherwise, pi/2 - result if x >= 1
    __m128 otherResult = _mm_sub_ps(PIDIV2, result);
    result = _mm_blendv_ps(result, otherResult, blendV);

    //force PI/2 if required
    result = _mm_blendv_ps(result, PIDIV2, forcePIDIV2);

    //factor in the sign
    return _mm_mul_ps(result, signMult);
}

__m128 arccos(__m128 x)
{
    const __m128 PIDIV2 = _mm_set1_ps(SEML_PI/2);
    return _mm_sub_ps(PIDIV2, arcsin(x));
}

__m128 arctan(__m128 x)
{
    //https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
    //solve for [-1, 1]
    //adjust input to solve for x >= 1
    const __m128 A = _mm_set1_ps(-0.00749305860992f);
    const __m128 B = _mm_set1_ps(0.03252232640125f);
    const __m128 C = _mm_set1_ps(-0.08467922817644f);
    const __m128 D = _mm_set1_ps(0.33288950512027f);
    const __m128 E = _mm_set1_ps(1.0f);
    const __m128 PIDIV2 = _mm_set1_ps(SEML_PI/2);
    
    __m128 signMult = _mm_blendv_ps(_mm_set1_ps(1), _mm_set1_ps(-1), _mm_cmplt_ps(x, _mm_set1_ps(0))); //multiply by the result
    x = abs(x);

    __m128 blendV = _mm_cmpgt_ps(x, _mm_set1_ps(1));
    x = _mm_blendv_ps(x, _mm_rcp_ps(x), blendV);

    __m128 xSqr = sqr(x);
    __m128 divisor = _mm_mul_ps(A, xSqr);
    divisor = _mm_mul_ps(xSqr, _mm_add_ps(B, divisor));
    divisor = _mm_mul_ps(xSqr, _mm_add_ps(C, divisor));
    divisor = _mm_mul_ps(xSqr, _mm_add_ps(D, divisor));
    divisor = _mm_add_ps(E, divisor);

    //Correct result for [-1, 1]
    __m128 result = _mm_div_ps(x, divisor);

    //otherwise, pi/2 - result if x >= 1
    __m128 otherResult = _mm_sub_ps(PIDIV2, result);
    result = _mm_blendv_ps(result, otherResult, blendV);

    return _mm_mul_ps(result, signMult);
}

__m128 arccsc(__m128 x)
{
    return arcsin(_mm_rcp_ps(x));
}

__m128 arcsec(__m128 x)
{
    return arccos(_mm_rcp_ps(x));
}

__m128 arccot(__m128 x)
{
    const __m128 PIDIV2 = _mm_set1_ps(SEML_PI/2);
    return _mm_sub_ps(PIDIV2, arctan(x));
}

__m128 sinh(__m128 x)
{
    __m128 numerator = _mm_sub_ps(exp(x), exp(negate(x)));
    return _mm_mul_ps(numerator, _mm_set1_ps(0.5));
}

__m128 cosh(__m128 x)
{
    __m128 numerator = _mm_add_ps(exp(x), exp(negate(x)));
    return _mm_mul_ps(numerator, _mm_set1_ps(0.5));
}

__m128 tanh(__m128 x)
{
    __m128 expValue = exp(_mm_mul_ps(x, _mm_set1_ps(2)));
    __m128 numerator = _mm_sub_ps(expValue, _mm_set1_ps(1));
    __m128 denominator = _mm_add_ps(expValue, _mm_set1_ps(1));
    return _mm_div_ps(numerator, denominator);
}

__m128 sech(__m128 x)
{
    __m128 denominator = _mm_add_ps(exp(x), exp(negate(x)));
    return _mm_div_ps(_mm_set1_ps(0.5), denominator);
}

__m128 csch(__m128 x)
{
    __m128 denominator = _mm_sub_ps(exp(x), exp(negate(x)));
    return _mm_div_ps(_mm_set1_ps(0.5), denominator);
}

__m128 coth(__m128 x)
{
    __m128 expValue = exp(_mm_mul_ps(x, _mm_set1_ps(2)));
    __m128 numerator = _mm_add_ps(expValue, _mm_set1_ps(1));
    __m128 denominator = _mm_sub_ps(expValue, _mm_set1_ps(1));
    return _mm_div_ps(numerator, denominator);
}

__m128 arcsinh(__m128 x)
{
    __m128 value = sqrt(_mm_add_ps(x, _mm_add_ps(sqr(x), _mm_set1_ps(1))));
    return ln(value);
}

__m128 arccosh(__m128 x)
{
    __m128 value = sqrt(_mm_add_ps(x, _mm_sub_ps(sqr(x), _mm_set1_ps(1))));
    return ln(value);
}

__m128 arctanh(__m128 x)
{
    __m128 numerator = _mm_add_ps(_mm_set1_ps(1), x);
    __m128 denominator = _mm_sub_ps(_mm_set1_ps(1), x);
    return _mm_mul_ps(_mm_set1_ps(0.5), ln(_mm_div_ps(numerator, denominator)));
}

__m128 arccsch(__m128 x)
{
    return arcsinh(_mm_rcp_ps(x));
}

__m128 arcsech(__m128 x)
{
    return arccosh(_mm_rcp_ps(x));
}

__m128 arccoth(__m128 x)
{
    __m128 numerator = _mm_add_ps(x, _mm_set1_ps(1));
    __m128 denominator = _mm_sub_ps(x, _mm_set1_ps(1));
    return _mm_mul_ps(_mm_set1_ps(0.5), ln(_mm_div_ps(numerator, denominator)));
}

