Title: 5th place solution with code | Kaggle

URL Source: https://www.kaggle.com/competitions/asl-signs/writeups/in-the-prize-line-5th-place-solution-with-code

Markdown Content:
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.

5th place solution with code — Google - Isolated Sign Language Recognition

Here is a quick overview of the 5th-place solution.

1.   **we applied various augmentations like flip, concatenation, etc**  
 1.1. By applying different augmentations, we can increase the cv by ~ 0.02 (0.76 -> 0.78)

2.   **the model is only a transformer model based on the public kernels**  
 2.1. By increasing the number of parameters, the performance of a single model can be increased to around 0.8 (0.78->0.8) in public LB.  
 2.1.1. 3 layers of transformer with the embedding size 480.

3.   **Preprocessing by mean and std of single sign sequence**  
 3.1. the preprocessing does affect the final performance.   
 3.1.1. we tried different ways of calculating the mean and std and found out that using the mean and std of the single sign sequence results in better cv.

4.   **Feature engineering like distances between points**  
 4.1. we selected and used around 106 points (as the public notebook by Heck).  
 4.2. distances withinpoints of hands/nose/eyes/… are calculated.

5.   **some methods to prevent overfitting like awp, random mask of frames, ema, etc …**

many thanks to my teammates [@qiaoshiji](https://www.kaggle.com/qiaoshiji)[@zengzhaoyang](https://www.kaggle.com/zengzhaoyang)

The source code for training models can be found here : [https://github.com/zhouyuanzhe/kaggleasl5thplacesolution](https://github.com/zhouyuanzhe/kaggleasl5thplacesolution)

Authors

## Citation

Qsj, zengzhy5, yuanzhe zhou. 5th place solution with code. https://www.kaggle.com/competitions/asl-signs/writeups/in-the-prize-line-5th-place-solution-with-code. 2023. Kaggle

## 1 Comment

Hotness

[](https://www.kaggle.com/akomwins)

感谢大佬分享code，已加🌟
