#pragma once
#include <immintrin.h>
#include <utility>

#ifndef SEML_PI
    #define SEML_PI 3.14159265359
#endif
#ifndef SEML_PI2
    #define SEML_PI2 6.28318530718
#endif
#ifndef SEML_E
	#define SEML_E 2.71828182846
#endif


namespace SEML
{
    /**
     * @brief Converts doubles to a 64 bit number. Does not work with all possible values.
     *      Range [0, 2^52] for unsigned. [-2^51, 2^51] for signed.
     *      Faster as it is 2 instructions
     *      https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
     * @param x 
     * @return __m256i 
     */
    __m256i fastDoubleToInt64(__m256d x);
    __m256i fastDoubleToUInt64(__m256d x);

    /**
     * @brief Converts a 64 bit number to a double. Does not work with all possible values.
     *      Range [0, 2^52] for unsigned. [-2^51, 2^51] for signed.
     *      Faster as it is 2 instructions
     *      https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
     * @param x 
     * @return __m256d 
     */
    __m256d fastInt64ToDouble(__m256i x);
    __m256d fastUInt64ToDouble(__m256i x);

    /**
     * @brief Converts a 64 bit number to a double. Works across the entire range
     *      https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
     * @param x 
     * @return __m256d 
     */
    __m256d int64ToDouble(__m256i x);
    __m256d uint64ToDouble(__m256i x);

    /**
     * @brief Reduces x to the range [-PI, PI]
     *      for trignometric functions
     * 
     * @param x 
     * @return __m256 
     */
    __m256 piRangeReduction(__m256 x);
    __m256d piRangeReduction(__m256d x);

    /**
     * @brief Converts radians to degrees
     * 
     * @param x 
     * @return __m256 
     */
    __m256 radToDeg(__m256 x);
    __m256d radToDeg(__m256d x);

    /**
     * @brief Converts degrees to radians
     * 
     * @param x 
     * @return __m256 
     */
    __m256 degToRad(__m256 x);
    __m256d degToRad(__m256d x);

    /**
     * @brief Computes the absolute value of x.
     *      Does so by flipping the sign bit in the floats.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 abs(__m256 x);
    __m256d abs(__m256d x);

    /**
     * @brief Computes the negative of x.
     *      Does not use multiplication so it is a bit faster.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 negate(__m256 x);
    __m256d negate(__m256d x);

    /**
     * @brief Extracts the sign of x as a value that can
     *      be multiplied back in after using abs(x).
     *      I.E.
     *          sign(x)*abs(x) = x
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sign(__m256 x);
    __m256d sign(__m256d x);

    /**
     * @brief Computes 1/x
     *      specifically computes _mm_div_ps(_mm_set1_ps(1), x)
     *      The fast version that is less accurate uses _mm_rcp_ps()
     * @param a 
     * @return __m256 
     */
    __m256 reciprocal(__m256 a);
    __m256d reciprocal(__m256d a);
    __m256 fastReciprocal(__m256 a);
    
    /**
     * @brief Computes the square of x.
     *      Is equivalent to _mm_mul_ps(x, x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sqr(__m256 x);
    __m256d sqr(__m256d x);
    
    /**
     * @brief Computes the cube of x.
     * 
     * @param a 
     * @return __m256 
     */
	__m256 cube(__m256 a);
	__m256d cube(__m256d a);

    /**
     * @brief Computes the Square Root of x.
     *      Is equivalent to _mm_sqrt_ps(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sqrt(__m256 x);
    __m256d sqrt(__m256d x);

    /**
     * @brief Computes the inverse sqrt of x.
     * @param a 
     * @return __m256 
     */
    __m256 invSqrt(__m256 a);
    __m256d invSqrt(__m256d a);

    /**
     * @brief Computes the inverse sqrt of x.
     *      While faster, it is a little less accurate.
     * @param a 
     * @return __m256 
     */
    __m256 fastInvSqrt(__m256 a);

    /**
     * @brief Computes the cos close to 0 with relatively high accuracy.
     *      used in the computation of sin and cos for single precision floating points
     * 
     * @param x 
     * @return __m256d 
     */
    __m256d cosAround0(__m256d x);

    /**
     * @brief Computes both sin and cos of the given angle.
     *      This is what is internally used for both sin and cos functions
     *      so this comes at no additional cost.
     * 
     *      Approach Used:
     *          https://www.youtube.com/watch?v=hffgNRfL1XY
     *              Skip to chapter 4
     * 
     *      Returns a pair where
     *          first = sin
     *          second = cos
     * 
     * @param x 
     * @return std::pair<__m256, __m256> 
     */
    std::pair<__m256, __m256> sincos(__m256 x);
    std::pair<__m256d, __m256d> sincos(__m256d x);

    /**
     * @brief Approximates the sine of x.
     *      Maintains the proper relationship between sine and cosine of x.
     *          cos(x) = sqrt(1 - sin(x)^2)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sin(__m256 x);
    __m256d sin(__m256d x);

    /**
     * @brief Approximates the cosine of x.
     *      Maintains the proper relationship between sine and cosine of x.
     *          cos(x) = sqrt(1 - sin(x)^2)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 cos(__m256 x);
    __m256d cos(__m256d x);

    /**
     * @brief Approximates the tangent of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 tan(__m256 x);
    __m256d tan(__m256d x);

    /**
     * @brief Approximates the secant of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sec(__m256 x);
    __m256d sec(__m256d x);

    /**
     * @brief Approximates the cosecant of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 csc(__m256 x);
    __m256d csc(__m256d x);

    /**
     * @brief Approximates the cotangent of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 cot(__m256 x);
    __m256d cot(__m256d x);

    /**
     * @brief Approximates the natural log of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 ln(__m256 x);
    __m256d ln(__m256d x);

    /**
     * @brief Approximates the natural log of x for values close to 1.
     *      Internally is used to approximate ln(x) for all x
     * 
     * @param x 
     * @return __m256d 
     */
    __m256d lnAround1(__m256d x);

    /**
     * @brief Approximates the log base 2 of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 log2(__m256 x);
    __m256d log2(__m256d x);

    /**
     * @brief Approximates the log base 10 of x.
     * 
     * @param x 
     * @return __m256 
     */
    __m256 log(__m256 x);
    __m256d log(__m256d x);

    /**
     * @brief Approximates the log of x with the specified base.
     * 
     * @param x 
     * @param base 
     * @return __m256 
     */
    __m256 log(__m256 x, float base);
    __m256d log(__m256d x, double base);

    /**
     * @brief Approximates the log of x with the specified base.
     * 
     * @param x 
     * @param base 
     * @return __m256 
     */
    __m256 log(__m256 x, __m256 base);
    __m256d log(__m256d x, __m256d base);

    /**
     * @brief Approximates e^x
     * 
     * @param x 
     * @return __m256 
     */
    __m256 exp(__m256 x);
    __m256d exp(__m256d x);

    /**
     * @brief Approximates e^x for values close to the origin 0
     *      Internally is used to approximate exp(x) for all x
     *      Uses double precision for more accuracy
     * 
     * @param x 
     * @return __m256d 
     */
    __m256d expAround0(__m256d x);

    /**
     * @brief Approximates x to the specified power
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 pow(__m256 x, float power);
    __m256d pow(__m256d x, double power);

    /**
     * @brief Approximates x to the specified power
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 pow(__m256 x, __m256 power);
    __m256d pow(__m256d x, __m256d power);


    /**
     * @brief Approximates the arcsine (sine inverse) of x
     *      Note that the inverse is only valid from [-1, 1] and returns the result in radians
     * 
     *      Approach used:
     *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 arcsin(__m256 x);
    __m256d arcsin(__m256d x);

    /**
     * @brief Approximates the arccosine (cosine inverse) of x
     *      Note that the inverse is only valid from [-1, 1] and returns the result in radians
     * 
     *      Approach used:
     *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 arccos(__m256 x);
    __m256d arccos(__m256d x);

    /**
     * @brief Approximates the arctan of x for values 1 or larger.
     *      Uses doubles for higher precision. For values less than 1, it diverges by a lot.
     *      accurate to 7 decimal places at 1. Higher accuracy as x increases.
     * 
     * @param x 
     * @return __m256d 
     */
    __m256d arctanApproxHigherThan1(__m256d x);

    /**
     * @brief Approximates the arctangent (tangent inverse) of x
     *      Returns the result in radians
     *      
     *      Approach used:
     *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 arctan(__m256 x);
    __m256d arctan(__m256d x);

    /**
     * @brief Approximates the arc cosecant (cosecant inverse) of x
     *      Note that the inverse is only valid from x <= -1 && x >= 1 and returns the result in radians
     *          Invalid in range [-1, 1]
     *      Approach used:
     *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 arccsc(__m256 x);
    __m256d arccsc(__m256d x);

    /**
     * @brief Approximates the arc secant (secant inverse) of x
     *      Note that the inverse is only valid from x <= -1 && x >= 1 and returns the result in radians
     *          Invalid in range [-1, 1]
     * 
     *      Approach used:
     *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 arcsec(__m256 x);
    __m256d arcsec(__m256d x);

    /**
     * @brief Approximates the arc cotangent (cotangent inverse) of x
     *      Returns the result in radians
     *      
     *      Approach used:
     *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
     * 
     * @param x 
     * @param power 
     * @return __m256 
     */
    __m256 arccot(__m256 x);
    __m256d arccot(__m256d x);

    /**
     * @brief Approximates the hyperbolic sine of x.
     *      Relies on the accuracy of exp(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sinh(__m256 x);
    __m256d sinh(__m256d x);

    /**
     * @brief Approximates the hyperbolic cosine of x.
     *      Relies on the accuracy of exp(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 cosh(__m256 x);
    __m256d cosh(__m256d x);

    /**
     * @brief Approximates the hyperbolic tangent of x.
     *      Relies on the accuracy of exp(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 tanh(__m256 x);
    __m256d tanh(__m256d x);

    /**
     * @brief Approximates the hyperbolic secant of x.
     *      Relies on the accuracy of exp(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 sech(__m256 x);
    __m256d sech(__m256d x);

    /**
     * @brief Approximates the hyperbolic cosecant of x.
     *      Relies on the accuracy of exp(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 csch(__m256 x);
    __m256d csch(__m256d x);

    /**
     * @brief Approximates the hyperbolic cotangent of x.
     *      Relies on the accuracy of exp(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 coth(__m256 x);
    __m256d coth(__m256d x);

    /**
     * @brief Approximates the inverse hyperbolic sine of x.
     *      Relies on the accuracy of ln(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 arcsinh(__m256 x);
    __m256d arcsinh(__m256d x);

    /**
     * @brief Approximates the inverse hyperbolic cosine of x.
     *      Relies on the accuracy of ln(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 arccosh(__m256 x);
    __m256d arccosh(__m256d x);

    /**
     * @brief Approximates the inverse hyperbolic tangent of x.
     *      Relies on the accuracy of ln(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 arctanh(__m256 x);
    __m256d arctanh(__m256d x);

    /**
     * @brief Approximates the inverse hyperbolic secant of x.
     *      Relies on the accuracy of ln(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 arcsech(__m256 x);
    __m256d arcsech(__m256d x);

    /**
     * @brief Approximates the inverse hyperbolic cosecant of x.
     *      Relies on the accuracy of ln(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 arccsch(__m256 x);
    __m256d arccsch(__m256d x);

    /**
     * @brief Approximates the inverse hyperbolic cotangent of x.
     *      Relies on the accuracy of ln(x)
     * 
     * @param x 
     * @return __m256 
     */
    __m256 arccoth(__m256 x);
    __m256d arccoth(__m256d x);
}