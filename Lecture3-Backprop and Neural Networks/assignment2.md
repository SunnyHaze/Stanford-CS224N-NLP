# 简答题部分
1. 求证naive-softmax损失函数与直接计算$\hat{y}$和$y$的交叉熵是否一致. 
$$y_w=
\begin{cases}
    0 & w \neq o \\
    1 & w = o
\end{cases}
$$
的条件下有：
$$
-\sum_{w\in \text{Vocab}}y_w\log(\hat{y_w}) = y_w\log (\hat{y}) = -\log (\hat{y_o}) 
$$
左侧是交叉熵，右侧是naive-softmax损失函数。

2. 具体的演算过程如下：
   $$
   \begin{aligned}
        \frac{\partial}{\partial \pmb{v}_c}(\pmb{J}_{\text{naive\_softmax})}(\pmb{v}_c,o,\pmb{U} )&= 
        \frac{\partial}{\partial \pmb{v}_c}\left(-\log P(O=o|C=c) \right)
        \\         
        &=\frac{\partial}{\partial \pmb{v}_c} 
        \left(-\log \frac{\exp(\pmb{u}_o^T \cdot \pmb{v}_c)}{\sum_{w\in \text{Vocab}}\exp({\pmb{u}_w^T\cdot \pmb{v}_c)}}\right) 
        \\
        &=        \frac{\partial}{\partial \pmb{v}_c} \left(-\log \cdot \exp(\pmb{u}_o^T \cdot \pmb{v}_c) +\log\left(\sum_{w\in V} \exp (\pmb{u}_w^T\cdot \pmb{v}_c )\right) \right) \\
        &=\frac{\partial}{\partial \pmb{v}_c} 
        \left(-\pmb{u}_0^T\cdot \pmb{v}_c\right) + 
        \frac{1}{\sum_{w\in V}\exp (\pmb{u}_w^T\cdot \pmb{v}_c)}\cdot
        \frac{\partial}{\partial \pmb{v}_c} \left(\sum_{x\in V}\exp(\pmb{u}_x^T \cdot \pmb{v}_c) \right)   
        \\
      &= -\pmb{u_o} +
              \sum_{x\in V} \frac{\exp(\pmb{u}_x^T\cdot \pmb{v}_c)}{\sum_{w\in V}\exp (\pmb{u}_w^T\cdot \pmb{v}_c)}\cdot \pmb{u}_x
        \\
        &=-\pmb{u}_o + \sum_{x\in V}P(O=x|C = c)\cdot \pmb{u}_x
        \\
        &=-\pmb{u}_o + \sum_{x\in V}\pmb{u}_x^T \cdot \hat{\pmb{y}}
        \\
        &=-\pmb{U}^T\pmb{y} +\pmb{U}^T \cdot \hat{\pmb{y}}
        \\
        &= \pmb{U}^T \cdot (\hat{\pmb{y}} - \pmb{y})
   \end{aligned}
   $$

3. 具体的演算过程如下：
   1. $w =o$ 时：
   $$
   \begin{aligned}
   \frac{\partial}{\partial \pmb{u}_w} (\pmb{J}_{\text{naive\_softmax})}(\pmb{v}_c,o,\pmb{U} )\ &= 
      \frac{\partial}{\partial \pmb{u}_w}\left(-\log(P(O=o|C=c))\right)
      \\
      &= \frac{\partial}{\partial \pmb{u}_w} \left(-\log \frac{\exp(\pmb{u}_o^T \cdot \pmb{v}_c)}{\sum_{w\in \text{Vocab}}\exp({\pmb{u}_w^T\cdot \pmb{v}_c)}} \right)
      \\
      &=    \frac{\partial}{\partial \pmb{u}_w}\left(-\log\cdot \exp(\pmb{u}_o^T\cdot \pmb{v}_c) \right) + \frac{\partial}{\partial \pmb{u}_w} \log \sum_{x\in V} \exp(\pmb{u}_x^T \cdot \pmb{v}_c)
      \\
      &= \frac{\partial}{\partial \pmb{u}_o}\left(-\pmb{u}_o^T \cdot \pmb{v}_c \right) + 
      \frac{1}{ \sum_{\in V} \exp(\pmb{u}_w^T \cdot \pmb{v}_c)} \cdot
      \frac{\partial}{\partial \pmb{u}_o}  \sum_{x\in V} \exp(\pmb{u}_x^T \cdot \pmb{v}_c)
      \\
      &=-\pmb{v}_c + \frac{1}{ \sum_{w\in V} \exp(\pmb{u}_w^T \cdot \pmb{v}_c)} \cdot \left(\exp(\pmb{u}_o^T \cdot \pmb{v}_c)\cdot \pmb{v}_c \right)
      \\
      &= -\pmb{v}_c + \frac{\exp(\pmb{u}_o^T \cdot \pmb{v}_c)}{ \sum_{w\in V} \exp(\pmb{u}_w^T \cdot \pmb{v}_c)} \cdot \pmb{v}_c
      \\
      &= -\pmb{v}_c + P(O = o| C = c) \cdot \pmb{v}_c
      \\
      &= \pmb{v}_c \cdot (\hat{y}_o - 1)
   \end{aligned}
   $$
   2. $w\neq o$ 时：
      1. 
      $$
      \begin{aligned}
          \frac{\partial}{\partial \pmb{u}_w} (\pmb{J}_{\text{naive\_softmax})}(\pmb{v}_c,o,\pmb{U} )\ &= 
         \frac{\partial}{\partial \pmb{u}_w}\left(-\log(P(O=o|C=c))\right) 
         \\
         &= \frac{\partial}{\partial \pmb{u}_w}\left(-\log\cdot \exp(\pmb{u}_o^T\cdot \pmb{v}_c) \right) + \frac{\partial}{\partial \pmb{u}_w} \log \sum_{x\in V} \exp(\pmb{u}_x^T \cdot \pmb{v}_c) 
         \\
         &= 0 \,+ \,  
          \frac{\partial}{\partial \pmb{u}_w} \exp(
         \pmb{u}_w^T \cdot \pmb{v}_c ) \cdot \frac{1}{\sum_{x\in V}\exp(\pmb{u}_x^T \cdot \pmb{v}_c)}
         \\
         &= 
          \frac{\exp(\pmb{u}_w^T \cdot \pmb{v}_c)}{\sum_{x\in V}\exp(\pmb{u}_x^T \cdot \pmb{v}_c)} \cdot  \pmb{v}_c
         \\
         &= 
         P(O=w|C = c) \cdot \pmb{v}_c 
         \\
         &=\hat{y} \cdot \pmb{v}_c
      \end{aligned}
      $$
      2. 将上述答案矩阵化，求解$\frac{\partial \pmb{J}(\pmb{v}_c,o,\pmb{U})}{\partial U}$的表达形式。
         $$
         \begin{aligned}
           \frac{\partial \pmb{J}(\pmb{v}_c,o,\pmb{U})}{\partial U} 
           &= \begin{bmatrix}
              \frac{\partial \pmb{J}(\pmb{v}_c,o,\pmb{U})}{\partial \pmb{u}_1}&\frac{\partial \pmb{J}(\pmb{v}_c,o,\pmb{U})}{\partial \pmb{u}_2} & {...} &
            \frac{\partial \pmb{J}(\pmb{v}_c,o,\pmb{U})}{\partial \pmb{u}_\text{Vocab}}
           \end{bmatrix}
         \end{aligned}
         $$ 
      3. Sigmoid 函数的导数计算
         $$
         \sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}
         $$
         $$
         \begin{aligned}
             \sigma'(x) &= -\frac{1}{(1+e^{-x})^2} \cdot (1+e^{-x})' 
             \\
             &= \frac{-e^{-x}}{1+ 2e^x + e^{2x}}
             \\
             &= \frac{1}{e^x + 2 + e^{-x}}
             \\
             &= \frac{1}{\frac{\sigma(x)}{1-\sigma(x)} + 2 + \frac{1 -\sigma(x)}{\sigma(x)}  }
             \\
             &=
             \frac{1-\sigma(x)}{\sigma(x) + 2(1-\sigma(x)) + \frac{1-2\sigma(x) + \sigma^2(x)}{\sigma(x)}}
             \\
             &=
             \frac{1-\sigma(x)}{\frac{1}{\sigma(x)}} 
             \\
             &=
             \ \sigma(x) (1-\sigma(x))
         \end{aligned}
         $$
            这个性质使得Sigmoid函数具有良好的求导结论
      4. 计算负采样的时候，重新计算偏导数
         $$
         J_\text{neg-sample}(\pmb{v}_c,o,\pmb{U})=-\log(\sigma(\pmb{u}_o^T\cdot \pmb{v}_c)) - \sum_{k=1}^k \log (\sigma(-\pmb{u}_k^T\cdot \pmb{v}_c))
         $$
         $$
         
         $$
      6. 