#include "SEML.h"

#include <iostream>
#include <functional>
#include <vector>
#include <math.h>

#define SIZE 0x100 //must be divisible by 4

//should probably use catch2 unit testing or any other unit test framework
void testFunction(std::function<float(float)> baseFunction, std::function<__m128(__m128)> sseFunction, std::vector<float>& testValues)
{
    float maxAbsError = 0;
    float minAbsError = INFINITY; //should never be this high
    float maxRelError = 0;
    float minRelError = INFINITY; //should never be this high
    float expectedValues[4];
    float sseValues[4];
    
    for(int i=0; i<SIZE; i+=4)
    {
        expectedValues[0] = baseFunction(testValues[i]);
        expectedValues[1] = baseFunction(testValues[i+1]);
        expectedValues[2] = baseFunction(testValues[i+2]);
        expectedValues[3] = baseFunction(testValues[i+3]);
        
        _mm_storeu_ps(sseValues, sseFunction(_mm_loadu_ps(&testValues[i])));

        //compare
        for(int k=0; k<4; k++)
        {
            float err = sseValues[k] - expectedValues[k];
            float absErr = abs(err);
            minAbsError = __min(absErr, minAbsError);
            maxAbsError = __max(absErr, maxAbsError);
            // printf("%d : f(%.9f) -> %.9f vs %.9f = %.9f\n", i+k, testValues[i+k], sseValues[k], expectedValues[k], err);

            float relativeErr = abs(err / expectedValues[k]);
            if(expectedValues[k] != 0)
            {
                minRelError = __min(relativeErr, minRelError);
                maxRelError = __max(relativeErr, maxRelError);
            }
        }
    }

    printf("\tMIN ABSOLUTE ERROR: %.9f\n", minAbsError);
    printf("\tMAX ABSOLUTE ERROR: %.9f\n", maxAbsError);
    printf("\tMIN RELATIVE ERROR: %.9f\n", minRelError);
    printf("\tMAX RELATIVE ERROR: %.9f\n", maxRelError);
}

void fillLinearRange(std::vector<float>& values, float a, float b)
{
    float range = b-a;
    for(int i=0; i<values.size(); i++)
    {
        values[i] = a + ((range*i)/SIZE);
    }
}

void testTrignometricFunctions(std::vector<float>& values)
{
    //values between -2pi and 2pi
    fillLinearRange(values, -SEML_PI2, SEML_PI2);
    printf("Testing sin(x):\n");
    testFunction([](float x)->float{return sin(x);}, [](__m128 x)->__m128{return SEML::sin(x);}, values);
    
    printf("Testing cos(x):\n");
    testFunction([](float x)->float{return cos(x);}, [](__m128 x)->__m128{return SEML::cos(x);}, values);
}

void testInverseTrignometricFunctions(std::vector<float>& values)
{
    //values between -1 and 1
    fillLinearRange(values, -1, 1);
    printf("Testing arcsin(x):\n");
    testFunction([](float x)->float{return asin(x);}, [](__m128 x)->__m128{return SEML::arcsin(x);}, values);
    
    //values between -10 and 10
    fillLinearRange(values, -10, 10);
    printf("Testing arctan(x):\n");
    testFunction([](float x)->float{return atan(x);}, [](__m128 x)->__m128{return SEML::arctan(x);}, values);
}

void testExpFunctions(std::vector<float>& values)
{
    //values between 0.1 and 10
    fillLinearRange(values, 0.01, 1);
    printf("Testing ln(x):\n");
    testFunction([](float x)->float{return log(x)/log(SEML_E);}, [](__m128 x)->__m128{return SEML::ln(x);}, values);
    
    //values between -10 and 10
    fillLinearRange(values, -10, 10);
    printf("Testing exp(x):\n");
    testFunction([](float x)->float{return exp(x);}, [](__m128 x)->__m128{return SEML::exp(x);}, values);
    
    //values between -10 and 10
    fillLinearRange(values, 0, 10);
    printf("Testing pow(x, 2.712):\n");
    testFunction([](float x)->float{return pow(x, 2.712);}, [](__m128 x)->__m128{return SEML::pow(x, 2.712f);}, values);
}

int main()
{
    //only testing functions that are newly added. Those are approximations
    //and not provided by SSE
    std::vector<float> values = std::vector<float>(SIZE);
    testTrignometricFunctions(values);
    testInverseTrignometricFunctions(values);
    testExpFunctions(values);
    return 0;
}