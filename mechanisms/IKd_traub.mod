TITLE Delayer rectifier

COMMENT
-----------------------------------------------------------------------------
	动作电位的延迟整流钾电流 / "delayer-rectifier" K current for action potentials
	---------------------------------------------------

  电压依赖性钾电流 / - potassium current, voltage-dependent
  迭代方程 / - iterative equations

  海马锥体细胞IKd模型，来源如下 / Model of IKd for hippocampal pyramidal cells, from
  参考文献：Traub与Miles，《海马神经元网络》，剑桥，1991年 / Traub & Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991

  作者Alain Destexhe，拉瓦尔大学，1996年 / Written by Alain Destexhe, Laval University, 1996
-----------------------------------------------------------------------------
ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX ikdT
	USEION k READ ek WRITE ik
	RANGE gkbar, vtraub
	RANGE n_inf
	RANGE tau_n
	RANGE n_exp
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gkbar	= .005 	(mho/cm2)
	vtraub	= -55	(mV)		: 调整阈值 / adjusts threshold
	ek	= -90	(mV)
	celsius = 36    (degC)
	dt              (ms)
	v               (mV)
}

STATE {
	n
}

ASSIGNED {
	ik	(mA/cm2)
	n_inf
	tau_n
	n_exp
	tadj
}


BREAKPOINT {
	SOLVE states
	ik  = gkbar * n*n*n*n * (v - ek)
}


: 已注释的状态变量微分方程块 / DERIVATIVE states {
: 已注释的电压依赖函数计算 / evaluate_fct(v)
: 已注释的n门控变量微分方程 / n' = (n_inf - n) / tau_n
:}

PROCEDURE states() {	: 电压v保持恒定时为精确解 / exact when v held constant
	evaluate_fct(v)
	n = n + n_exp * (n_inf - n)
}

UNITSOFF
INITIAL {
:
: 两种电流的Q10均假设为2.3 / Q10 was assumed to be 2.3 for both currents
:
: 原始测量在室温下进行 / original measurements at room temperature

	tadj = 3.0 ^ ((celsius-36)/ 10 )
	evaluate_fct(v)
	n = n_inf
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2

	v2 = v - vtraub : 转换为Traub约定 / convert to traub convention

	a = 0.032 * (15-v2) / ( exp((15-v2)/5) - 1)
	b = 0.5 * exp((10-v2)/40)

	tau_n = 1 / (a + b) / tadj
	n_inf = a / (a + b)

	n_exp = 1 - exp(-dt/tau_n)
}

UNITSON
