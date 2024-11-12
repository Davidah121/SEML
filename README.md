# SEML - SIMD Extended Math Library
A simple library that approximates important math functions in SSE.

SVML is an intel extension to SSE/AVX/AVX-512 which is not supported by every compiler limited the usability of SSE.

SEML is a cross platform option that does not rely on Intel only code and is relatively simple to move to AVX, AVX-512, or any other SIMD based instruction set.

# Metrics
Many functions attempt to have low absolute error where possible aiming for accuracy of 6 decimal places.

All base functions (sin, cosine, exp, ln, arctan) acheive 6 or more decimal places of accuracy with respect to absolute error.

All functions are based on the accuracy of those base functions so in order to improve the other function's accuracy, improvement of the fundamental functions is required. (pow for instance needs high accuracy in exp and ln)
