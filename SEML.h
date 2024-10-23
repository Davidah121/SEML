#pragma once
#include <immintrin.h>

#ifndef SEML_PI
    #define SEML_PI 3.141593 //3.14159265359
#endif
#ifndef SEML_PI2
    #define SEML_PI2 6.283186
#endif
#ifndef SEML_E
	#define SEML_E 2.718282
#endif

/**
 * @brief Reduces x to the range [-PI, PI]
 *      for trignometric functions
 * 
 * @param x 
 * @return __m128 
 */
__m128 piRangeReduction(__m128 x);

/**
 * @brief Converts radians to degrees
 * 
 * @param x 
 * @return __m128 
 */
__m128 radToDeg(__m128 x);

/**
 * @brief Converts degrees to radians
 * 
 * @param x 
 * @return __m128 
 */
__m128 degToRad(__m128 x);

/**
 * @brief Computes the absolute value of x.
 *      Does so by flipping the sign bit in the floats.
 * 
 * @param x 
 * @return __m128 
 */
__m128 abs(__m128 x);

/**
 * @brief Computes the negative of x.
 *      Does not use multiplication so it is a bit faster.
 * 
 * @param x 
 * @return __m128 
 */
__m128 negate(__m128 x);

/**
 * @brief Computes the square of x.
 *      Is equivalent to _mm_mul_ps(x, x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 sqr(__m128 x);

/**
 * @brief Computes the Square Root of x.
 *      Is equivalent to _mm_sqrt_ps(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 sqrt(__m128 x);

/**
 * @brief Approximates the sine of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 sin(__m128 x);

/**
 * @brief Approximates the cosine of x.
 *      Maintains the proper relationship between sine and cosine of x.
 *          cos(x) = sqrt(1 - sin(x)^2)
 * 
 * @param x 
 * @return __m128 
 */
__m128 cos(__m128 x);

/**
 * @brief Approximates the tangent of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 tan(__m128 x);

/**
 * @brief Approximates the secant of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 sec(__m128 x);

/**
 * @brief Approximates the cosecant of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 csc(__m128 x);

/**
 * @brief Approximates the cotangent of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 cot(__m128 x);

/**
 * @brief Approximates the natural log of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 ln(__m128 x);

/**
 * @brief Approximates the natural log of x for values close to 1.
 *      Internally is used to approximate ln(x) for all x
 * 
 * @param x 
 * @return __m128 
 */
__m128 lnAround1(__m128 x);

/**
 * @brief Approximates the log base 2 of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 log2(__m128 x);

/**
 * @brief Approximates the log base 10 of x.
 * 
 * @param x 
 * @return __m128 
 */
__m128 log(__m128 x);

/**
 * @brief Approximates the log of x with the specified base.
 * 
 * @param x 
 * @param base 
 * @return __m128 
 */
__m128 log(__m128 x, float base);

/**
 * @brief Approximates the log of x with the specified base.
 * 
 * @param x 
 * @param base 
 * @return __m128 
 */
__m128 log(__m128 x, __m128 base);

/**
 * @brief Approximates e^x
 * 
 * @param x 
 * @return __m128 
 */
__m128 exp(__m128 x);

/**
 * @brief Approximates 2^x for values close to the origin 0
 *      Internally is used to approximate exp(x) for all x
 * 
 * @param x 
 * @return __m128 
 */
__m128 exp2Around0(__m128 x);

/**
 * @brief Approximates x to the specified power
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 pow(__m128 x, float power);

/**
 * @brief Approximates x to the specified power
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 pow(__m128 x, __m128 power);


/**
 * @brief Approximates the arcsine (sine inverse) of x
 *      Note that the inverse is only valid from [-1, 1] and returns the result in radians
 * 
 *      Approach used:
 *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 arcsin(__m128 x);

/**
 * @brief Approximates the arccosine (cosine inverse) of x
 *      Note that the inverse is only valid from [-1, 1] and returns the result in radians
 * 
 *      Approach used:
 *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 arccos(__m128 x);

/**
 * @brief Approximates the arctangent (tangent inverse) of x
 *      Returns the result in radians
 *      
 *      Approach used:
 *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 arctan(__m128 x);

/**
 * @brief Approximates the arc cosecant (cosecant inverse) of x
 *      Note that the inverse is only valid from x <= -1 && x >= 1 and returns the result in radians
 *          Invalid in range [-1, 1]
 *      Approach used:
 *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 arccsc(__m128 x);

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
 * @return __m128 
 */
__m128 arcsec(__m128 x);

/**
 * @brief Approximates the arc cotangent (cotangent inverse) of x
 *      Returns the result in radians
 *      
 *      Approach used:
 *          https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482
 * 
 * @param x 
 * @param power 
 * @return __m128 
 */
__m128 arccot(__m128 x);

/**
 * @brief Approximates the hyperbolic sine of x.
 *      Relies on the accuracy of exp(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 sinh(__m128 x);

/**
 * @brief Approximates the hyperbolic cosine of x.
 *      Relies on the accuracy of exp(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 cosh(__m128 x);

/**
 * @brief Approximates the hyperbolic tangent of x.
 *      Relies on the accuracy of exp(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 tanh(__m128 x);

/**
 * @brief Approximates the hyperbolic secant of x.
 *      Relies on the accuracy of exp(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 sech(__m128 x);

/**
 * @brief Approximates the hyperbolic cosecant of x.
 *      Relies on the accuracy of exp(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 csch(__m128 x);

/**
 * @brief Approximates the hyperbolic cotangent of x.
 *      Relies on the accuracy of exp(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 coth(__m128 x);

/**
 * @brief Approximates the inverse hyperbolic sine of x.
 *      Relies on the accuracy of ln(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 arcsinh(__m128 x);

/**
 * @brief Approximates the inverse hyperbolic cosine of x.
 *      Relies on the accuracy of ln(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 arccosh(__m128 x);

/**
 * @brief Approximates the inverse hyperbolic tangent of x.
 *      Relies on the accuracy of ln(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 arctanh(__m128 x);

/**
 * @brief Approximates the inverse hyperbolic secant of x.
 *      Relies on the accuracy of ln(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 arcsech(__m128 x);

/**
 * @brief Approximates the inverse hyperbolic cosecant of x.
 *      Relies on the accuracy of ln(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 arccsch(__m128 x);

/**
 * @brief Approximates the inverse hyperbolic cotangent of x.
 *      Relies on the accuracy of ln(x)
 * 
 * @param x 
 * @return __m128 
 */
__m128 arccoth(__m128 x);
