# 基于 MPC 的路径跟踪

> \(x=[p_x,p_y,v_x,v_y]^T\)，输入为 \(u=[a_x,a_y]^T\)。

## 1. 离散双积分器模型

连续模型为
\[
\dot p=v,\qquad \dot v=a.
\]
以采样周期 \(T_s=2.01\) s 作零阶保持离散化（常加速度模型），得到
\[
x_{k+1}=Ax_k+Bu_k,
\quad A=\begin{bmatrix}1&0&T_s&0\\0&1&0&T_s\\0&0&1&0\\0&0&0&1\end{bmatrix},
\quad B=\begin{bmatrix}T_s^2/2&0\\0&T_s^2/2\\T_s&0\\0&T_s\end{bmatrix}.
\]
这是线性时不变系统，因此在给定初始状态后，整个预测轨迹是输入序列的仿射函数。

## 2. 参考轨迹与代价函数

道路中心线由 waypoint 插值得到，记采样点为 \(r_i=[r_{x,i},r_{y,i}]^T\)。参考状态有两种：

* **Task 1/Task 2：** 仅跟踪位置，\(\bar x_i=[r_{x,i},r_{y,i},0,0]^T\)。
* **Task 3：** 速度采用前向差分，\(\bar v_i=(r_{i+1}-r_i)/T_s\)，闭合轨迹最后一点用首点（或 `circshift`）连接。

取
\[
Q=\operatorname{diag}(1,1,0,0),\qquad R=2.2\times10^3I_2,
\]
预测长度为一圈采样点数 \(N=T\)。MPC 目标为
\[
J=\sum_{i=0}^{N-1}\left[(x_i-\bar x_i)^TQ(x_i-\bar x_i)+u_i^TRu_i\right]
 +(x_N-\bar x_N)^TQ(x_N-\bar x_N).
\]
较大的 \(R\) 抑制剧烈加速度，使轨迹平滑；若跟踪偏差过大，可适当增大位置权重。

## 3. 将 MPC 写成标准 QP

令 \(U=[u_0^T,\ldots,u_{N-1}^T]^T\)，堆叠状态 \(X=[x_0^T,\ldots,x_N^T]^T\)。由系统递推可写成
\[
X=\mathcal A x_0+\mathcal B U,
\]
其中第 \(i\) 个块行满足 \(x_i=A^ix_0+\sum_{j=0}^{i-1}A^{i-1-j}Bu_j\)。令 \(e=\mathcal A x_0-\bar X\)，\(\bar X\) 为堆叠参考，\(\bar Q=\operatorname{blkdiag}(Q,\ldots,Q)\)，\(\bar R=I_N\otimes R\)，则
\[
J=\tfrac12U^THU+f^TU+\text{常数},\quad
H=2(\mathcal B^T\bar Q\mathcal B+\bar R),\quad
f=2\mathcal B^T\bar Qe.
\]
\(R\succ0\) 保证 \(H\succ0\)，所以 QP 解唯一。

## 4. Task 1：无约束 MPC

问题为 \(\min_U \frac12U^THU+f^TU\)。解析解
\[
U^*=-H^{-1}f
\]
（实际实现使用线性方程求解 `H\\(-f)`，不要显式求逆）。每个仿真时刻执行“求解--施加第一个输入--状态更新--滚动参考”的 receding horizon 循环。无约束时应能较好跟踪中心线，但弯道处可能出现过大加速度。

## 5. Task 2：加入输入约束

加入逐分量约束
\[
u_{\min}\le u_i\le u_{\max},\qquad u_{\min}=(-1,-1)^T,\;u_{\max}=(1,1)^T.
\]
等价于 `lb <= U <= ub` 的凸 QP。使用 MATLAB `quadprog(H,f,[],[],[],[],lb,ub)`，或项目中的投影梯度法：每次梯度更新后执行
\[
U\leftarrow\Pi_{[lb,ub]}(U)=\min(ub,\max(lb,U)).
\]
输入饱和会使弯道跟踪误差增大，这是约束导致的正常现象；可增加预测长度或降低参考速度改善可行性。

## 6. Task 3：输入约束与道路边界约束

对每个采样点，以内外边界连线构造线性不等式。代码中的
\[
k_i=-\frac{o_{x,i}-\iota_{x,i}}{o_{y,i}-\iota_{y,i}},\qquad
c_i=[-k_i,\;1,\;0,\;0]^T,
\]
并令
\[
d_i^{\min}=\min(\iota_{y,i}-k_i\iota_{x,i},\;o_{y,i}-k_io_{x,i}),\quad
d_i^{\max}=\max(\cdots),
\]
即可得到
\[
d_i^{\min}\le c_i^Tx_i\le d_i^{\max}.
\]
堆叠后，由 \(X=\mathcal A x_0+\mathcal B U\) 转为输入空间约束：
\[
d_{\min}-C\mathcal A x_0\le C\mathcal B U\le d_{\max}-C\mathcal A x_0,
\]
其中 \(C=\operatorname{blkdiag}(c_0^T,\ldots,c_N^T)\)。与输入上下界一起交给 QP 求解器。若某时刻 QP 不可行，应降低参考速度、放宽边界（考虑车辆宽度），或采用软约束
\(d_{\min}-s\le CX\le d_{\max}+s,\;s\ge0\)，并在目标中加入 \(\rho\|s\|_2^2\)。

## 7. 仿真与结果检查

每步只施加最优序列的第一个输入，并使用含噪声的实际状态作为下一步估计值；记录 \(p_x,p_y,v_x,v_y\)。应绘制：道路内外边界、红色参考中心线、蓝色实际轨迹，并检查：

1. 所有输入均满足 \([-1,1]\)；
2. 轨迹点满足每个边界不等式；
3. 位置误差和输入变化平滑，无明显发散；
4. 闭合轨迹首尾连续（最后参考速度使用首点差分）。

综上，三个任务分别展示了无约束、输入约束、输入加状态约束下的凸二次规划 MPC；约束只改变可行域，不改变问题的凸性和唯一最优解性质。
