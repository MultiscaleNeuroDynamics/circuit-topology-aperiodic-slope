TITLE Sodium channels

COMMENT
-----------------------------------------------------------------------------
	动作电位的钠电流 / Na current for action potentials
	--------------------------------

  快速钠电流 / - fast sodium current
  迭代方程 / - iterative equations

  海马锥体细胞INa模型，来源如下 / Model of INa for hippocampal pyramidal cells, from
  参考文献：Traub与Miles，《海马神经元网络》，剑桥，1991年 / Traub & Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991

  添加仅作用于失活的偏移参数 / Added a shift parameter for inactivation only

  作者Alain Destexhe，拉瓦尔大学，1996年 / Written by Alain Destexhe, Laval University, 1996
-----------------------------------------------------------------------------
ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX inaT
	USEION na READ ena WRITE ina
	RANGE gnabar, vtraub, shift
	RANGE m_inf, h_inf
	RANGE tau_m, tau_h
	RANGE m_exp, h_exp
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gnabar	= .003 	(mho/cm2)	: 最大电导 / max conductance
	vtraub	= -65	(mV)		: 调整阈值 / adjusts threshold
	shift	= 0	(mV)		: 失活偏移 / inactivation shift
	ena	= 50	(mV)
	celsius = 36    (degC)
	dt              (ms)
	v               (mV)
}

STATE {
	m h
}

ASSIGNED {
	ina	(mA/cm2)
	m_inf
	h_inf
	tau_m	(ms)
	tau_h	(ms)
	m_exp
	h_exp
	tadj
}


BREAKPOINT {
	SOLVE states
	ina = gnabar * m*m*m*h * (v - ena)
}


: 已注释的状态变量微分方程块 / DERIVATIVE states {
: 已注释的电压依赖函数计算 / evaluate_fct(v)
: 已注释的m门控变量微分方程 / m' = (m_inf - m) / tau_m
: 已注释的h门控变量微分方程 / h' = (h_inf - h) / tau_h
:}

PROCEDURE states() {	: 电压v保持恒定时为精确解 / exact when v held constant
	evaluate_fct(v)
	m = m + m_exp * (m_inf - m)
	h = h + h_exp * (h_inf - h)
}

UNITSOFF
INITIAL {
	m = 0
	h = 0
:
: 两种电流的Q10均假设为2.3 / Q10 was assumed to be 2.3 for both currents
:
: 原始测量在室温下进行 / original measurements at room temperature

	tadj = 2.3 ^ ((celsius-23)/ 10 )
	evaluate_fct(v)
	m = m_inf
	h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2

	v2 = v - vtraub 	: 转换为Traub约定 / convert to traub convention

	a = 0.32 * (13-v2) / ( exp((13-v2)/4) - 1)
	b = 0.28 * (v2-40) / ( exp((v2-40)/5) - 1)
	tau_m = 1 / (a + b)
	m_inf = a / (a + b)

	v2 = v2 - shift		: 失活偏移 / inactivation shift

	a = 0.128 * exp((17-v2)/18)
	b = 4 / ( 1 + exp((40-v2)/5) )
	tau_h = 1 / (a + b)
	h_inf = a / (a + b)

	m_exp = 1 - exp(-dt/tau_m)
	h_exp = 1 - exp(-dt/tau_h)
}

UNITSON
