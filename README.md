# The Kolmogorov-Smirnov Test - Lab

## Introduction
In the previous lesson, we saw that the Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution function of the sample and the cumulative distribution function of the reference distribution, or between the empirical distribution functions of two samples. In this lab, we shall see how to perform this test in Python. 

## Objectives

In this lab you will:

- Calculate a one- and two-sample Kolmogorov-Smirnov test
- Interpret the results of a one- and two-sample Kolmogorov-Smirnov test
- Compare K-S test to visual approaches for testing for normality assumption

### Data

Let's import the necessary libraries and generate some data. Run the following cell: 


```python
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

np.random.seed(42)
# Create the normal random variables with mean 0, and sd 3
x_10 = stats.norm.rvs(loc=0, scale=3, size=10)
x_50 = stats.norm.rvs(loc=0, scale=3, size=50)
x_100 = stats.norm.rvs(loc=0, scale=3, size=100)
x_1000 = stats.norm.rvs(loc=0, scale=3, size=1000)
```

### Plots 

Plot histograms and Q-Q plots of above datasets and comment on the output 

- How good are these techniques for checking normality assumptions?
- Compare both these techniques and identify their limitations/benefits etc. 



```python
datasets = ['x_10', 'x_50', 'x_100', 'x_1000']
for ind, i in enumerate([x_10, x_50, x_100, x_1000]):
    print(datasets[ind])
    plt.hist(i)
    sm.qqplot(i, line='s')
    plt.show();
```

    x_10



![png](index_files/index_5_1.png)



![png](index_files/index_5_2.png)


    x_50



![png](index_files/index_5_4.png)



![png](index_files/index_5_5.png)


    x_100



![png](index_files/index_5_7.png)



![png](index_files/index_5_8.png)


    x_1000



![png](index_files/index_5_10.png)



![png](index_files/index_5_11.png)



```python
# Plot histograms and Q-Q plots for above datasets


```

# Your comments here 
As dataset gets larger, so does the normal distribution / qq look more normal.

### Create a function to plot the normal CDF and ECDF for a given dataset
- Create a function to generate an empirical CDF from data
- Create a normal CDF using the same mean = 0 and sd = 3, having the same number of values as data


```python
plt.plot(np.sort(x_10), np.linspace(0, 1, len(x_10), endpoint=False))
plt.plot(np.sort(stats.norm.rvs(loc=0, scale=3, size=len(x_10))), np.linspace(0, 1, len(x_10), endpoint=False))

plt.legend(['ECDF', 'CDF'])
plt.title('Comparing CDFs for K-S test, Sample size=' + str(len(x_10)))
```




    Text(0.5, 1.0, 'Comparing CDFs for K-S test, Sample size=10')




![png](index_files/index_9_1.png)



```python
# You code here 

def ks_plot(data):
    plt.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))
    plt.plot(np.sort(stats.norm.rvs(loc=0, scale=3, size=len(data))), np.linspace(0, 1, len(data), endpoint=False))

    plt.legend(['ECDF', 'CDF'])
    plt.title('Comparing CDFs for K-S test, Sample size=' + str(len(data)))
    plt.show();
    pass
    
# Uncomment below to run the test
ks_plot(stats.norm.rvs(loc=0, scale=3, size=100)) 
ks_plot(stats.norm.rvs(loc=5, scale=4, size=100))
```


![png](index_files/index_10_0.png)



![png](index_files/index_10_1.png)


This is awesome. The difference between the two CDFs in the second plot shows that the sample did not come from the distribution which we tried to compare it against. 

Now you can run all the generated datasets through the function `ks_plot()` and comment on the output.


```python
for ind, i in enumerate([x_10, x_50, x_100, x_1000]):
    ks_plot(i);
```


![png](index_files/index_12_0.png)



![png](index_files/index_12_1.png)



![png](index_files/index_12_2.png)



![png](index_files/index_12_3.png)



```python
# Your code here 
```


```python
# Your comments here 
```

### K-S test in SciPy

Let's run the Kolmogorov-Smirnov test, and use some statistics to get a final verdict on normality. We will test the hypothesis that the sample is a part of the standard t-distribution. In SciPy, we run this test using the function below:

```python
scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='approx')
```
Details on arguments being passed in can be viewed at this [link to the official doc.](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html)

Run the K-S test for normality assumption using the datasets created earlier and comment on the output: 
- Perform the K-S test against a normal distribution with mean = 0 and sd = 3
- If p < .05 we can reject the null hypothesis and conclude our sample distribution is not identical to a normal distribution 


```python
# Perform K-S test 
np.random.seed(999)
for i in [x_10, x_50, x_100, x_1000]:
    print (stats.kstest(i, 'norm', args=(0, 3)))
# KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
# KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
# KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
# KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)
for i in [x_10, x_50, x_100, x_1000]:
    print (stats.kstest(i, 'norm', args=(25, 5)))
```

    KstestResult(statistic=0.3193652946234613, pvalue=0.20920466940428856)
    KstestResult(statistic=0.1735707056389606, pvalue=0.0869033638768608)
    KstestResult(statistic=0.07881202004760868, pvalue=0.548454773294235)
    KstestResult(statistic=0.03333275499708793, pvalue=0.211976526032451)
    KstestResult(statistic=0.9999746603882038, pvalue=2.1828785211724866e-46)
    KstestResult(statistic=0.9999495948075252, pvalue=2.65946248553582e-215)
    KstestResult(statistic=0.9997858921376609, pvalue=0.0)
    KstestResult(statistic=0.9981910812597394, pvalue=0.0)


### Your comments here 
The datasets have a p value greater than 0.05 and shows that they are of the same distribution. Changing the normal distribution to have a different loc and sd does give us a p-value < 0.05 demonstrating that we can reject the hypothesis that the samples are from teh same distribution.

Generate a uniform distribution and plot / calculate the K-S test against a uniform as well as a normal distribution: 


```python
x_uni = np.random.rand(1000)
# Try with a uniform distribution
print(stats.kstest(x_uni, lambda x: x))
print(stats.kstest(x_uni, 'norm', args=(0,3)))
# KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
# KstestResult(statistic=0.5000553288071681, pvalue=0.0)
```

    KstestResult(statistic=0.03523781819690697, pvalue=0.16301839932805795)
    KstestResult(statistic=0.5000285749629829, pvalue=9.973485866132198e-232)


# Your comments here 
First is not significant, the second is, meaning the datasets did not come from the same distribution. 

## Two-sample K-S test

A two-sample K-S test is available in SciPy using following function: 

```python 
scipy.stats.ks_2samp(data1, data2)[source]
```

Let's generate some bi-modal data first for this test: 


```python
# Generate binomial data
N = 1000
x_1000_bi = np.concatenate((np.random.normal(-1, 1, int(0.1 * N)),
                            np.random.normal(5, 1, int(0.9 * N))))[:, np.newaxis]
plt.hist(x_1000_bi);
```


![png](index_files/index_24_0.png)



```python
len(x_1000), len(x_1000_bi)
```




    (1000, 1000)



Plot the CDFs for `x_1000_bimodal` and `x_1000` and comment on the output. 


```python
# Plot the CDFs
def ks_plot_2sample(data_1, data_2):
    '''
    Data entered must be the same size.
    '''
    if len(data_1) == len(data_2):
        plt.plot(np.sort(data_1), np.linspace(0, 1, len(data_1), endpoint=False))
        plt.plot(np.sort(data_2), np.linspace(0, 1, len(data_2), endpoint=False))
        plt.legend(labels=['Data_1', 'Data_2'])
        plt.title('Comparing 2 CDFs for KS-Test') 
        plt.show();
    pass

# Uncomment below to run
ks_plot_2sample(x_1000, x_1000_bi[:,0])

```


![png](index_files/index_27_0.png)


### You comments here 
KS test should come out different.

Run the two-sample K-S test on `x_1000` and `x_1000_bi` and comment on the results. 


```python
# Your code here
stats.ks_2samp(x_1000, x_1000_bi[:,0])
# Ks_2sampResult(statistic=0.633, pvalue=4.814801487740621e-118)
```




    Ks_2sampResult(statistic=0.717, pvalue=3.1733429211806902e-248)



### Your comments here 
Indeed p-value is less than 0.05, and the datasets come from different distributions.

## Summary

In this lesson, we saw how to check for normality (and other distributions) using one- and two-sample K-S tests. You are encouraged to use this test for all the upcoming algorithms and techniques that require a normality assumption. We saw that we can actually make assumptions for different distributions by providing the correct CDF function into Scipy K-S test functions. 
