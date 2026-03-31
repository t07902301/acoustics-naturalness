# Audio Evaluation Methods

## Introduction
This repository provides a custom implementation of audio evaluation methods, specifically focusing on the Mel-Frequency Cepstral Coefficients (MFCC) and Dynamic Time Warping (DTW) techniques. These methods are widely used in the field of audio signal processing for tasks such as speech recognition, audio classification, and more.

## Mel-Frequency Cepstral Coefficients (MFCC)
MFCCs are a representation of the short-term power spectrum of a sound. They are commonly used in speech and audio processing because they effectively capture the timbral aspects of audio signals. 

## Dynamic Time Warping (DTW)
DTW is a dynamic-programming-based algorithm for measuring discrepancy between two temporal sequences that may vary in speed. It's useful for aligning audio signals that may have been recorded at different speeds or have varying lengths. Here's an outline of how DTW works:

### Key Steps in DTW:
1. **Cost Matrix Initialization**: Create a matrix to store the cumulative cost.
2. **Cost Calculation**: Fill in the matrix based on the distance between points.
3. **Optimal Path Calculation**: Find the path through the matrix that minimizes the cumulative cost.
4. **Warping Path**: Use the optimal path to align the sequences.