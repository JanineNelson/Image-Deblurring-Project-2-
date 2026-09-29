IMAGE DEBLURRING AND ILL-CONDITIONED OPTIMIZATION
Project 2

PROJECT OVERVIEW

This project studies how to recover a sharper image from a blurred and noisy observation. It examines why image deblurring is an ill-conditioned optimization problem and how Tikhonov regularization improves stability.

The full report, code, figures, and analysis are available in image_deblurring.md.

PROBLEM AND MOTIVATION

Blur makes it difficult to identify boundaries and small features in an image. This can affect tasks such as inspecting manufactured parts.

Reversing blur can sharpen an image, but it can also amplify measurement noise. This project explores that tradeoff using a controlled synthetic image.

EXPERIMENT SETUP

The experiment uses a 128 by 128 grayscale image containing a square, a disk, and thin bars.

A Gaussian filter blurs the reference image. Gaussian noise is then added to create the observed image.

Main settings:
- Gaussian blur width: 2 pixels
- Noise standard deviation: 0.03
- Random seed: 0
- Regularization coefficient: 0.01
- Relative-gradient stopping tolerance: 0.00001
- Maximum gradient-descent updates: 5,000

OPTIMIZATION APPROACH

The baseline method searches for an image whose blurred version matches the observed image. It minimizes the sum of squared differences using gradient descent.

The image intensities are continuous and unconstrained. Values are not clipped during optimization or error calculations.

Blur removes some image patterns much more strongly than others. This produces a large spread in Hessian eigenvalues, causing slow convergence and sensitivity to noise.

REQUIRED DIAGNOSTICS

D1: Spectrum
Examine the Hessian eigenvalues and estimate the condition number.

D2: Intrinsic ill-conditioning
Investigate how conditioning changes with blur width and whether diagonal scaling improves it.

D3: Effect on optimization
Measure baseline gradient-descent convergence using a fixed stopping tolerance.

D4: Proposed remedy
Compare the baseline with Tikhonov regularization using convergence curves and reconstruction error.

TIKHONOV REGULARIZATION

Tikhonov regularization adds a penalty on the squared image intensities to the original objective.

This improves conditioning and limits noise amplification. However, it also introduces bias and can reduce genuine detail.

Regularization changes the optimization problem. It is not simply a faster way to solve the original problem.

MAIN RESULTS

For the main experiment with a blur width of 2 pixels:

Baseline gradient descent:
- Reached the 5,000-update limit without meeting the tolerance.
- Reconstruction RMSE was approximately 0.3435.

Tikhonov gradient descent:
- Reached the tolerance in 369 updates.
- Reconstruction RMSE was approximately 0.0810.
- Regularized Hessian condition number was approximately 101.

A narrower blur of 0.5 pixels reached the same tolerance in 51 updates.

The very large unregularized condition estimate requires care because of floating-point precision. The full report explains this limitation.

REPOSITORY CONTENTS

image_deblurring.md
The complete report, Python code blocks, and recorded results.

image_deblurring_files/
Figures referenced by the report.

requirements.txt
Python dependencies.

settings.json
Existing project settings.

StreamLitApp 
an app that runs through deblurring process

REPRODUCIBILITY

The experiment generates its own synthetic image, so no external dataset is needed.

Install the packages listed in requirements.txt. Run the Python code blocks from the report in a notebook, in their displayed order.

Fixed random seeds make the experiments repeatable. Small numerical differences may occur across library versions.

LIMITATIONS

The model assumes known, uniform Gaussian blur and periodic image boundaries.

The experiments use synthetic grayscale images and additive Gaussian noise.

A small gradient does not guarantee an accurate reconstruction.

The results demonstrate the behavior of this controlled model. They do not establish general performance on real photographs or inspection images.
