# IFAC PROFESSIONAL BRIEF

# Computer Control: An Overview

**Björn Wittenmark**
http://www.control.lth.se/~bjorn

**Karl Johan Åström**
http://www.control.lth.se/~kja

**Karl-Erik Årzén**
http://www.control.lth.se/~karlerik

Department of Automatic Control
Lund Institute of Technology
Lund, Sweden

## Abstract

Computer control is entering all facets of life from home electronics to production of different products and material. Many of the computers are embedded and thus "hidden" for the user. In many situations it is not necessary to know anything about computer control or real-time systems to implement a simple controller. There are, however, many situations where the result will be much better when the sampled-data aspects of the system are taken into consideration when the controller is designed. Also, it is very important that the real-time aspects are regarded. The real-time system influences the timing in the computer and can thus minimize latency and delays in the feedback controller.

The paper introduces different aspects of computer-controlled systems from simple approximation of continuous time controllers to design aspects of optimal sampled-data controllers. We also point out some of the pitfalls of computer control and discusses the practical aspects as well as the implementation issues of computer control.

---

## Contents

1. [Introduction](#1-introduction)
2. [Sampling and Reconstruction](#2-sampling-and-reconstruction)
3. [Mathematical Models](#3-mathematical-models)
4. [Frequency Response](#4-frequency-response)
5. [Control Design and Specifications](#5-control-design-and-specifications)
6. [Approximation of Analog Controllers](#6-approximation-of-analog-controllers)
7. [Feedforward Design](#7-feedforward-design)
8. [PID Control](#8-pid-control)
9. [Pole-placement Design](#9-pole-placement-design)
10. [Optimization Based Design](#10-optimization-based-design)
11. [Practical Issues](#11-practical-issues)
12. [Real-time Implementation](#12-real-time-implementation)
13. [Controller Timing](#13-controller-timing)
14. [Research Issues](#14-research-issues)

---

## 1. Introduction

Computers are today essential for implementing controllers in many different situations. The computers are often used in embedded systems. An embedded system is a built-in computer/microprocessor that is a part of a larger system. Many of these computers implement control functions of different physical processes, for example, vehicles, home electronics, cellular telephones, and stand-alone controllers. The computers are often hidden for the end-user, but it is essential that the whole system is designed in an effective way.

Using computers has many advantages. Many problems with analog implementation can be avoided by using a computer, for instance, there are no problems with the accuracy or drift of the components. The computations are performed identically day after day. It is also possible to make much more complicated computations, such as iterations and solution of system of equations, using a computer. All nonlinear, and also many linear operations, using analog technique are subject to errors, while they are much more accurately made using a computer. Logic for alarms, start-up, and shut-down is easy to include in a computer. Finally, it is possible to construct good graphical user interfaces. However, sometimes the full advantages of the computers are not utilized. One situation is when the computer is only used to approximately implement an analog controller. The full potential of a computer-controlled system is only obtained when also the design process is regarded from a sampled-data point of view. The theory presented in this paper is essentially for linear deterministic systems. Even if the control algorithms give the same result for identical input data the performance of the closed-loop system crucially depends on the timing given by the real-time operating system. This paper gives a brief review of some of the tools that are important for understanding, analyzing, designing, and implementing sampled-data control systems.

### Computer-controlled Systems

A schematic diagram of a computer-controlled system is shown in Figure 1. The system consists of:

- **Process**
- **Sampler** together with Analog-to-Digital (A-D) converter
- **Digital-to-Analog (D-A) converter** with a hold circuit
- **Computer** with a clock, software for real-time applications, and control algorithms
- **Communication network**

The process is a continuous-time physical system to be controlled. The input and the output of the process are continuous-time signals. The A-D converter is converting the analog output signal of the process into a finite precision digital number depending on how many bits or levels that are used in the conversion. The conversion is also quantized in time determined by the clock. This is called the sampling process. The control algorithm thus receives data that are quantized both in time and in level. The control algorithm consists of a computer program that transforms the measurements into a desired control signal. The control signal is transfered to the D-A converter, which with finite precision converts the number in the computer into a continuous-time signal. This implies that the D-A converter contains both a conversion unit and a hold unit that translates a number into a physical variable that is applied to the process. The communication between the process, or more accurately the A-D and D-A converters, and the computer is done over a communication link or network. All the activities in the computer-controlled system are controlled by the clock with the requirement that all the computations have to be performed within a given time. In a distributed system there are several clocks that have to be synchronized. The total system is thus a real-time system with hard time constraints.

![Figure 1: Schematic diagram of a computer-controlled system](images/figure_01.png)

The system in Figure 1 contains a continuous-time part and a sampled-data part. It is this mixture of different classes of signals that causes some of the problems and confusion when discussing computer-controlled systems. The problem is even more pronounced when using decentralized control where the control system is shared between several computers. Many of the problems are avoided if only considering the sampled-data part of the system. It is, however, important to understand where the difficulties arise and how to avoid some of these problems.

### The Sampling Process

The times when the measured physical variables are converted into digital form are called the sampling instants. The time between two sampling instants is the sampling period and is denoted h. Periodic sampling is normally used, implying that the sampling period is constant, i.e. the output is measured and the control signal is applied each hth time unit. The sampling frequency is ωₛ = 2π/h. In larger plants it may also be necessary to have control loops with different sampling periods, giving rise to multi-rate sampling or multi-rate systems.

### Time Dependence

The mixture of continuous-time and discrete-time signals makes the sampled-data systems time dependent. This can be understood from Figure 1. Assume that there is a disturbance added at the output of the process. Time invariance implies that a shift of the input signal to the system should result in a similar shift in the response of the system. Since the A-D conversion is governed by a clock the system will react differently when the disturbance is shifted in time. The system will, however, remain time independent provided that all changes in the system, inputs and disturbances, are synchronized with the sampling instants.

### Aliasing and New Frequencies

The A-D and D-A converters are the interface between the continuous-time reality and the digital world inside the computer. A natural question to ask is: Do we loose any information by sampling a continuous-time signal?

**EXAMPLE 1—ALIASING**

Consider the two sinusoidal signals sin((1.8t - 1)π) and sin(0.2πt) shown in Figure 2. The sampling of these signals with the sampling period h = 1 is shown with the dots in the figure. From the sampled values there is thus no possibility to distinguish between the two signals. It is not possible to determine if the sampled values are obtained from a low frequency signal or a high frequency signal. This implies that the continuous-time signal cannot be recovered from the sampled-data signal, i.e. information is lost through the sampling procedure.

![Figure 2: Two signals with different frequencies, 0.1 Hz (dashed) and 0.9 Hz (full), have the same values at all sampling instants (dots) when h = 1](images/figure_02.png)

The sampling procedure does not only introduce loss of information but may also introduce new frequencies. The new frequencies are due to interference between the sampled continuous-time signal and the sampling frequency ωₛ = 2π/h. The interference can introduce fading or beating in the sampled-data signal.

### Approximation of Continuous-time Controllers

One way to design a sampled-data control system is to make a continuous-time design and then make a discrete-time approximation of this controller. The computer-controlled system should now behave as the continuous-time system provided the sampling period is sufficiently small. An example is used to illustrate this procedure.

**EXAMPLE 2—DISK-DRIVE POSITIONING SYSTEM**

A disk-drive positioning system will be used throughout the paper to illustrate different concepts. A schematic diagram is shown in Figure 3. Neglecting resonances the dynamics relating the position y of the arm to the voltage u of the drive amplifier is approximately described by a double integrator process:

$$G(s) = \frac{k}{Js^2}$$ (1)

where k is the gain in the amplifier and J is the moment of inertia of the arm. The purpose of the control system is to control the position of the arm so that the head follows a given track and that it can be rapidly moved to a different track. Let uₖ be the command signal and denote Laplace transforms with capital letters. A simple continuous-time servo controller can be described by:

$$U(s) = \frac{bK}{a}U_c(s) - K\frac{s + b}{s + a}Y(s)$$ (2)

![Figure 3: A system for controlling the position of the arm of a disk drive](images/figure_03.png)

This controller is a two-degrees-of-freedom controller. Choosing the controller parameters as:

- a = 2ω₀
- b = ω₀/2
- K = 2Jω₀²/k

gives a closed system with the characteristic polynomial:

$$P(s) = (s^2 + ω₀s + ω₀^2)(s + ω₀) = s^3 + 2ω₀s^2 + 2ω₀^2s + ω₀^3$$

where the parameter ω₀ is used to determine the speed of the closed-loop system. The step response when controlling the process with the continuous-time controller (2) is shown as the dashed line in Figure 4. The control law given by (2) can be written as:

$$U(s) = \frac{bK}{a}U_c(s) - KY(s) - K\frac{a-b}{s+a}Y(s) = K\left(\frac{b}{a}U_c(s) - Y(s) - X(s)\right)$$

or in the time-domain as:

$$u(t) = K\left(\frac{b}{a}u_c(t) - y(t) - x(t)\right)$$

$$\frac{dx(t)}{dt} = -ax(t) + (a-b)y(t)$$

![Figure 4: The step response when simulating the disk arm servo with the analog controller (2) (dashed/red) and sampled-data controller (3) (solid/blue). The design parameter is ω₀ = 1 and the sampling period is h = 0.3](images/figure_04.png)

The first of these equations can be implemented directly on a computer. To find an algorithm for the second equation the derivative dx/dt at time kh is approximated with a forward difference. This gives:

$$\frac{x(kh + h) - x(kh)}{h} = -ax(kh) + (a-b)y(kh)$$

The following approximation of the continuous algorithm (2) is then obtained:

$$u(kh) = K\left(\frac{b}{a}u_c(kh) - y(kh) - x(kh)\right)$$

$$x(kh + h) = x(kh) + h[(a-b)y(kh) - ax(kh)]$$ (3)

where y(kh) is the sampled output, u(kh) is the signal sent to the D-A converter, and x(kh) is the internal state of the controller. This difference equation is updated at each sampling instant. Since the approximation of the derivative by a difference is good if the interval h is small, we can expect the behavior of the computer-controlled system to be close to the continuous-time system. Figure 4 shows the arm position and the control signal for the system when ω₀ = 1 and h = 0.3. Notice that the control signal for the computer-controlled system is constant between the sampling instants due to the hold circuit in the D-A converter. Also notice that the difference between the outputs of the two simulations is very small. The difference in the outputs is negligible if h is less than 0.1/ω₀.

The computer-controlled system has slightly higher overshoot and the settling time is a little longer. The difference between the systems decreases when the sampling period decreases. When the sampling period increases the computer-controlled system will, however, deteriorate and the closed-loop system becomes unstable for long sampling periods.

The example shows that it is straightforward to obtain an algorithm for computer control simply by writing the continuous-time control law as a differential equation and approximating the derivatives by differences. The example also indicates that the procedure seems to work well if the sampling period is sufficiently small. The overshoot and the settling time are, however, a little larger for the computer-controlled system, i.e. there is a deterioration due to the approximation.

**Python Implementation using ctrlsys:**

```python
#!/usr/bin/env python3
"""Disk-Drive Positioning System Example using ctrlsys"""
import numpy as np
from ctrlsys import ab04md, tf01md
import matplotlib.pyplot as plt

def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space."""
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    den = den / den[0]
    num = num / den[0]
    n = len(den) - 1
    if n == 0:
        return (np.zeros((0, 0), order='F'), np.zeros((0, 1), order='F'),
                np.zeros((1, 0), order='F'), np.array([[num[0]]], order='F'))
    num_padded = np.zeros(n + 1)
    num_padded[n + 1 - len(num):] = num
    A = np.zeros((n, n), order='F', dtype=float)
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]
    B = np.zeros((n, 1), order='F', dtype=float)
    B[-1, 0] = 1.0
    C = np.zeros((1, n), order='F', dtype=float)
    d0 = num_padded[0]
    for i in range(n):
        C[0, n - 1 - i] = num_padded[i + 1] - d0 * den[i + 1]
    D = np.array([[d0]], order='F', dtype=float)
    return A, B, C, D

# Plant parameters
k, J = 1.0, 1.0  # Amplifier gain and moment of inertia
omega_0 = 1.0    # Design parameter (determines speed)
h = 0.3          # Sampling period

# Create continuous-time plant: G(s) = k/(J*s^2)
A_plant, B_plant, C_plant, D_plant = tf_to_ss([k], [J, 0, 0])
print(f"Plant: G(s) = {k}/{J}s²")

# Controller parameters (equation 2)
a = 2 * omega_0
b = omega_0 / 2
K = 2 * J * omega_0**2 / k

print(f"\nController parameters:")
print(f"  a = 2ω₀ = {a}")
print(f"  b = ω₀/2 = {b}")
print(f"  K = 2Jω₀²/k = {K}")

# Simulate continuous-time closed-loop using fine discretization
dt_cont = 0.01
A_d, B_d, C_d, D_d, _ = ab04md('C',
    np.asfortranarray(A_plant), np.asfortranarray(B_plant),
    np.asfortranarray(C_plant), np.asfortranarray(D_plant),
    alpha=1.0, beta=2.0/dt_cont)

n_cont = int(15 / dt_cont)
t_cont = np.arange(n_cont) * dt_cont
y_cont = np.zeros(n_cont)
x_cont = np.zeros(A_d.shape[0])
x_ctrl_cont = 0.0

for i in range(n_cont):
    y_k = (C_d @ x_cont + D_d.flatten() * 0).item()
    y_cont[i] = y_k
    u_k = K * (b/a * 1.0 - y_k - x_ctrl_cont)
    x_ctrl_cont += dt_cont * ((a - b) * y_k - a * x_ctrl_cont)
    x_cont = A_d @ x_cont + B_d.flatten() * u_k

# DISCRETE-TIME IMPLEMENTATION (equation 3)
class DiskDriveController:
    """Discrete controller for disk drive (equation 3)."""
    def __init__(self, K, a, b, h):
        self.K, self.a, self.b, self.h = K, a, b, h
        self.x = 0.0

    def calculate_output(self, uc, y):
        return self.K * (self.b/self.a * uc - y - self.x)

    def update_state(self, y):
        self.x = self.x + self.h * ((self.a - self.b) * y - self.a * self.x)

# Discrete plant model (Tustin approximation)
A_disc, B_disc, C_disc, D_disc, _ = ab04md('C',
    np.asfortranarray(A_plant), np.asfortranarray(B_plant),
    np.asfortranarray(C_plant), np.asfortranarray(D_plant),
    alpha=1.0, beta=2.0/h)

# Simulation
n_steps = int(15 / h)
t_disc = np.arange(n_steps) * h
y_disc = np.zeros(n_steps)
u_disc = np.zeros(n_steps)

controller = DiskDriveController(K, a, b, h)
x_plant = np.zeros(A_disc.shape[0])
setpoint = 1.0

for i in range(n_steps):
    y_k = (C_disc @ x_plant).item()
    y_disc[i] = y_k
    u_k = controller.calculate_output(setpoint, y_k)
    u_disc[i] = u_k
    controller.update_state(y_k)
    x_plant = A_disc @ x_plant + B_disc.flatten() * u_k

# Plot results (matching Figure 4)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(t_cont, y_cont, 'r--', label='Continuous (Analog)', linewidth=2)
ax1.plot(t_disc, y_disc, 'b-', label='Discrete (Sampled-data)', linewidth=1.5)
ax1.axhline(1, color='k', linestyle=':', alpha=0.3)
ax1.set_ylabel('Position y(t)')
ax1.set_title(f'Disk Drive Control: ω₀={omega_0}, h={h}')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.step(t_disc, u_disc, 'b-', where='post', label='Control signal', linewidth=1.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Control u(t)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('disk_drive_control.png')
print(f"\nPerformance: Discrete overshoot: {(max(y_disc) - 1.0)*100:.1f}%")
```

This implementation demonstrates:
1. **Plant model**: Double integrator G(s) = k/(Js²)
2. **Controller design**: Two-degrees-of-freedom controller (equation 2)
3. **Discretization**: Forward difference approximation (equation 3)
4. **Simulation**: Comparison of continuous vs. discrete control
5. **Verification**: Output matches Figure 4 behavior

### Is There a Need for Special Design Methods?

The approximation of a continuous-time controller discussed above shows that it is possible to derive a discrete-time controller based on continuous-time design. There are many different approximations methods that can be used. Normally, they are crucially dependent on choosing fairly short sampling periods. The approximation methods are possible to use also for nonlinear continuous-time controllers. Deriving a sampled-data description of the process makes it possible to utilize the full potential of a sampled-data systems and to derive other classes of controllers that are not possible to use in continuous-time. The sampled-data theory is also needed to explain the inter-sample and periodic behavior of the sampled-data systems. Notice that the theory for sampled-data systems is derived mainly for linear systems.

**EXAMPLE 3—DEADBEAT CONTROL**

Consider the disk drive in the previous example. Using a sampling interval of h = 1.4 with ω₀ = 1 gives an unstable closed-loop system if the controller (3) is used. However, designing a discrete-time controller with the same structure as (3) gives the performance shown in Figure 5 when h = 1.4. The controller can be written as:

$$u(kh) = t_0u_c(kh) + t_1u_c(kh - h) - s_0y(kh) - s_1y(kh - h) - r_1u(kh - h)$$

where the coefficients now are chosen such that the output of the system will reach the reference value after two samples. This is called a deadbeat controller and it has not a continuous-time counterpart. The design parameter of the deadbeat controller is the sampling period h, which here is chosen such that the maximum control signal is the same when the continuous-time and the sampled-data controllers are used.

![Figure 5: The step response when simulating the disk arm servo using the analog controller (3) (dashed/red) and the deadbeat controller (solid/blue). The sampling period is h = 1.4](images/figure_05.png)

### Summary

The examples in this chapter clearly show that there are some important issues that are specific for sampled-data systems.

- Sampling makes the system time-varying.
- Information may be lost through sampling.
- Sampled controller have behaviors that are not achievable with continuous-time controllers.

This requires an understanding of the interplay between the continuous-time signals and the discrete-time signals. The rest of the paper gives a brief overview of some of the tools that are important for the analysis, design, and implementation of sampled-data control system.

---

## 2. Sampling and Reconstruction

A sampler is a device, driven by the clock in the system, that is converting a continuous-time signal into a sequence of numbers. Normally, the sampler is combined into the A-D converter, which also quantizes the sequence of numbers into a finite, although it may be a high, precision number that is then stored in the computer. Typically the A-D converter has 8–16 bits resolution giving 2⁸–2¹⁶ levels of quantization. This is normally a much higher resolution than the precision of the physical sensor. The input to the process is reconstructed from the sequence of numbers from the computer using a D-A converter combined with a hold circuit that is determining the input to the process until a new number is delivered from the computer. The hold circuit can be of different kinds. Most common is the zero-order-hold, which holds the input constant over the sampling period. Another type of hold device that will be discussed is the first-order-hold, where the input is computed from current and previous outputs from the computer using a first order polynomial in time.

### Shannon's Sampling Theorem

It is clear that if the signal to be sampled is not changing very fast and if the sampling is done sufficiently often very little should be lost in the sampling procedure. This intuition was formalized by Claude E. Shannon in 1949 in his famous sampling theorem. Shannon proved that if the signal contains no frequencies above ω₀ then the continuous-time signal can be uniquely reconstructed from a periodically sampled sequence provided the sampling frequency is higher than 2ω₀.

In Figure 2 the sinusoidal signals has the frequencies 0.1 Hz and 0.9 Hz, respectively, while the sampling frequency is 1 Hz. From Shannon's sampling theorem it follows that the slow 0.1 Hz signal can be uniquely reconstructed from its sampled values. To be able to reconstruct the 0.9 Hz signal the sampling frequency must be higher than 1.8 Hz.

The Shannon reconstruction of the continuous-time signal from the sampled values is characterized by the filter:

$$h(t) = \frac{\sin(ω_st/2)}{ω_st/2}$$

The impulse response of this filter is given in Figure 6. The frequency ω_N = ω_s/2 plays an important role and is called the Nyquist frequency. The filter for the Shannon reconstruction is not causal, which makes it impossible to use in practice and simpler and less accurate reconstructions, such as the zero-order hold, are therefore used.

![Figure 6: The impulse response of the Shannon reconstruction when h = 1](images/figure_06.png)

### Aliasing and Antialiasing Filters

The phenomenon shown in Figure 2 that a high frequency signal will be interpreted as a low frequency signal when sampled is called aliasing or frequency folding. The fundamental alias for a frequency ω₁ is given by:

$$ω = |(ω_1 + ω_N) \mod (ω_s) - ω_N|$$ (4)

Equation (4) implies that the signal ω₁ has an alias in the interval [0, ω_N] and that the signal above the Nyquist frequency cannot be reconstructed after the sampling. Equation (4) gives that 0.9 Hz has the alias frequency 0.1 Hz when the sampling frequency is 1 Hz as could be seen in Figure 2.

The implication of the aliasing is that we need to remove all frequencies above the Nyquist frequency before sampling the signal. The simplest way of doing this is to use an analog filter. This type of filters are called antialiasing filters. The bandwidth of the filter must be such that the attenuation above the Nyquist frequency is sufficiently high. Bessel filters of orders 2–6 are in practice sufficient to eliminate most of the influence of higher frequencies. A second order Bessel filter with bandwidth ω_B has the transfer function:

$$\frac{ω_B^2}{(s/ω)^2 + 2ζω(s/ω) + ω^2}$$

with ω = 1.27 and ζ = 0.87. Other filters such as Butterworth or ITAE can also be used. The Bessel filters has the property that they can be well approximated by a time delay. This is an advantage in the design of the controller since the dynamics of the antialiasing filter normally has to be included in the design of the sampled-data controller.

**EXAMPLE 4—PREFILTERING**

The influence of a prefilter is shown in Figure 7. An analog signal consisting of a square-wave and a superimposed sinusoidal disturbance with frequency 0.9 Hz is shown in (a). In (c) the signal in (a) is sampled using a sampling frequency of 1 Hz. This gives rise to an alias frequency of 0.1 that is seen in the sampled signal. The result of filtering the signal in (a) using a sixth-order Bessel filter with a bandwidth of 0.25 Hz is shown in (b) and the sampling of this signal is given in (d). The disturbance signal is eliminated in (b), but the filter also has an influence on the "useful" signal, i.e. the square-wave is transformed into a smoother signal, which is sampled.

![Figure 7: (a) Signal plus sinusoidal disturbance. (b) The signal filtered through a sixth-order Bessel-filter. (c) Sampling of the signal in (a). (d) Sampling of the signal in (b)](images/figure_07.png)

### Zero-order Hold (ZOH)

The simplest and the most common way to make a reconstruction of a sampled-data signal is to let the output of the hold circuit be constant until the next sampled value is obtained. This is called zero-order hold, since the continuous time signal is a zeroth order polynomial between the sampling points. The reconstructed signal f(t) is given by:

$$f(t) = f(kh), \quad kh ≤ t < kh + h$$

Standard D-A converters are usually constructed such that the output is constant until a new conversion is ordered. The zero-order hold circuit can easily also be used for n-dimensional systems by having one hold circuit for each component.

The transfer function for the zero-order hold is:

$$H_0(s) = \frac{1-e^{-sh}}{s}$$

The frequency response is obtained by substituting s = iω:

$$H_0(iω) = \frac{1-e^{-iωh}}{iω} = he^{-iωh/2}\frac{\sin(ωh/2)}{ωh/2}$$

This gives the magnitude and phase:

$$|H_0(iω)| = h\left|\frac{\sin(ωh/2)}{ωh/2}\right|$$

$$\arg H_0(iω) = -\frac{ωh}{2}$$

The phase curve is linear in ω and can thus be interpreted as a time delay of h/2, which is the average of the time delay through the zero-order hold circuit. The amplitude curve shows that the zero-order hold circuit has the characteristics of a low-pass filter.

### First-order Hold

The zero-order hold is simple and gives reasonably good reconstruction. Other hold circuits may, however, give better performance. The first-order hold, also called triangle hold, gives an output that is a first-order polynomial. This implies that the first-order hold circuit extrapolates linearly based on the last two samples. The first-order hold can be written as:

$$f(t) = f(kh) + \frac{t-kh}{h}[f(kh) - f(kh-h)], \quad kh ≤ t < kh + h$$

The transfer function for the first-order hold is:

$$H_1(s) = \frac{(1+sh)(1-e^{-sh})}{sh}$$

The first-order hold circuit has some advantages over the zero-order hold. One is that the output is a continuous signal, another advantage is that the first-order hold gives a better approximation to the continuous-time signal. The first-order hold has, however, the disadvantage that it requires more computation since the previous value also has to be stored. Also, it introduces a phase lag of one sampling period, i.e. twice as large as for the zero-order hold. This may cause stability problems in closed-loop systems.

### Summary

The sampling and reconstruction are important steps in a sampled-data system. The following are key points:

- Shannon's sampling theorem gives the minimal sampling rate needed to reconstruct a signal.
- Aliasing occurs when the signal contains frequencies above the Nyquist frequency.
- Antialiasing filters are used to remove frequencies above the Nyquist frequency.
- Zero-order hold is the most common reconstruction method.
- First-order hold gives smoother reconstruction but introduces more delay.

---

## 3. Mathematical Models

To be able to analyze and design computer-controlled systems it is necessary to have good mathematical models. Different models are useful for different purposes. State-space models and input-output models are two common classes of models. This chapter discusses how to obtain discrete-time models starting from continuous-time models.

### Zero-order Hold Sampling of a Continuous-time System

Consider a continuous-time system described by the state-space model:

$$\frac{dx(t)}{dt} = Ax(t) + Bu(t)$$

$$y(t) = Cx(t) + Du(t)$$

Assume that the input signal is piecewise constant over the sampling intervals, i.e. the system has a zero-order hold at the input. The solution to the differential equation is:

$$x(t) = e^{A(t-t_0)}x(t_0) + \int_{t_0}^{t} e^{A(t-τ)}Bu(τ)dτ$$

Introduce t₀ = kh and t = kh + h. Since the control signal is constant during the sampling interval:

$$x(kh+h) = e^{Ah}x(kh) + \left(\int_0^h e^{Aτ}dτ\right)Bu(kh)$$

Introduce the notation:

$$Φ = e^{Ah}$$

$$Γ = \int_0^h e^{Aτ}dτ \cdot B = A^{-1}(e^{Ah}-I)B$$

The sampled system can then be described by:

$$x(kh+h) = Φx(kh) + Γu(kh)$$

$$y(kh) = Cx(kh) + Du(kh)$$

This is a discrete-time state-space model that describes the exact relationship between the input and output at the sampling instants. The model does not, however, give any information about the behavior between the sampling instants.

### First-order Hold Sampling

If a first-order hold circuit is used instead of the zero-order hold, the analysis becomes more complicated. The input signal is now:

$$u(t) = u(kh) + \frac{t-kh}{h}[u(kh) - u(kh-h)], \quad kh ≤ t < kh + h$$

The discrete-time state-space model becomes:

$$x(kh+h) = Φx(kh) + Γ_0u(kh) + Γ_1u(kh-h)$$

$$y(kh) = Cx(kh) + Du(kh)$$

where:

$$Γ_0 = \int_0^h e^{Aτ}\left(1+\frac{τ}{h}\right)dτ \cdot B$$

$$Γ_1 = -\int_0^h e^{Aτ}\frac{τ}{h}dτ \cdot B$$

The first-order hold sampling gives a model that depends on both the current and previous input values.

### Solution of the System Equation

The solution to the discrete-time state-space equation:

$$x(k+1) = Φx(k) + Γu(k)$$

$$y(k) = Cx(k) + Du(k)$$

is given by:

$$x(k) = Φ^kx(0) + \sum_{i=0}^{k-1}Φ^{k-1-i}Γu(i)$$

$$y(k) = CΦ^kx(0) + \sum_{i=0}^{k-1}CΦ^{k-1-i}Γu(i) + Du(k)$$

This shows that the output at time k depends on the initial state and all previous inputs.

### Operator Calculus and Input-output Descriptions

The forward shift operator q is defined by:

$$qy(k) = y(k+1)$$

Using this operator, the state-space equation can be written as:

$$qx(k) = Φx(k) + Γu(k)$$

$$y(k) = Cx(k) + Du(k)$$

Solving for x(k):

$$x(k) = (qI - Φ)^{-1}Γu(k)$$

gives:

$$y(k) = [C(qI - Φ)^{-1}Γ + D]u(k)$$

This is the pulse-transfer function of the system:

$$H(q) = C(qI - Φ)^{-1}Γ + D$$

### Z-transform

The z-transform is a fundamental tool for analyzing discrete-time systems. For a sequence {f(k)}, k = 0, 1, 2, ..., the z-transform is defined as:

$$F(z) = \sum_{k=0}^{\infty}f(k)z^{-k}$$

The z-transform has many properties similar to the Laplace transform. Table 1 summarizes the most important properties:

**Table 1: Some Properties of the Z-transform**

| Property | Formula |
|----------|---------|
| **1. Definition** | $$F(z) = \sum_{k=0}^{\infty}f(kh)z^{-k}$$ |
| **2. Inversion** | $$f(kh) = \frac{1}{2\pi i}\oint F(z)z^{k-1}dz$$ |
| **3. Linearity** | $$Z[af + bg] = aZ[f] + bZ[g]$$ |
| **4. Time shift** | $$Z[q^{-n}f] = z^{-n}F(z)$$ <br> $$Z[q^n f] = z^n(F(z) - F_1(z))$$ where $$F_1(z) = \sum_{j=0}^{n-1}f(jh)z^{-j}$$ |
| **5. Initial-value theorem** | $$f(0) = \lim_{z→\infty}F(z)$$ |
| **6. Final-value theorem** | If $$(1-z^{-1})F(z)$$ has no poles on or outside the unit circle, then <br> $$\lim_{k→\infty}f(kh) = \lim_{z→1}(1-z^{-1})F(z)$$ |
| **7. Convolution** | $$Z[f * g] = Z\left[\sum_{n=0}^{k}f(n)g(k-n)\right] = F(z)G(z)$$ |

### The Pulse-transfer Function

The pulse-transfer function relates the z-transforms of the output and input:

$$Y(z) = H(z)U(z)$$

where:

$$H(z) = C(zI - Φ)^{-1}Γ + D$$

The pulse-transfer function can also be written as a rational function:

$$H(z) = \frac{B(z)}{A(z)} = \frac{b_0z^n + b_1z^{n-1} + ... + b_n}{z^n + a_1z^{n-1} + ... + a_n}$$

The poles of the system are the roots of A(z) = 0, and the zeros are the roots of B(z) = 0.

### Zero-order Hold Sampling

For a continuous-time system with transfer function G(s), the pulse-transfer function with zero-order hold sampling is:

$$H(z) = (1-z^{-1})Z\left[\frac{G(s)}{s}\right]$$

This formula is very useful for computing pulse-transfer functions. Table 2 gives the pulse-transfer functions for some common continuous-time systems with zero-order hold sampling.

**Table 2: Zero-order Hold Sampling of Common Transfer Functions**

| G(s) | H(z) |
|------|------|
| $$\frac{1}{s}$$ | $$\frac{hz}{z-1}$$ |
| $$\frac{1}{s^2}$$ | $$\frac{h^2(z+1)}{2(z-1)^2}$$ |
| $$\frac{a}{s+a}$$ | $$\frac{1-e^{-ah}}{z-e^{-ah}}$$ |
| $$\frac{1}{s(s+a)}$$ | $$\frac{ah-1+e^{-ah}}{a(z-1)(z-e^{-ah})}$$ |
| $$\frac{ω_0^2}{s^2+2ζω_0s+ω_0^2}$$ | Complex expression (see references) |

**Example:**

For G(s) = 1/(s+a):

$$H(z) = (1-z^{-1})Z\left[\frac{1}{s(s+a)}\right] = (1-z^{-1})\frac{z}{(z-1)(z-e^{-ah})} = \frac{1-e^{-ah}}{z-e^{-ah}}$$

### First-Order-Hold Sampling

For first-order hold sampling, the pulse-transfer function becomes more complicated and includes a delay:

$$H(z) = z^{-1}H_1(z)$$

where H₁(z) can be computed using z-transform tables.

### Shift-operator Calculus and Z-transforms

The shift operator q and the z-transform variable z are related by q = z. This means that results derived using the shift operator can be directly translated to z-transform expressions by replacing q with z.

### Summary

This chapter has introduced mathematical models for sampled-data systems:

- State-space models for zero-order and first-order hold sampling
- The shift operator and z-transform
- Pulse-transfer functions
- Methods for computing discrete-time models from continuous-time models

These models are essential tools for analysis and design of sampled-data control systems.

---

## 4. Frequency Response

Frequency response is an important tool for analyzing control systems. This chapter discusses how frequency response is defined for sampled-data systems and how it differs from continuous-time systems.

### Propagation of Sinusoidal Signals

Consider a stable discrete-time system with pulse-transfer function H(z). If the input is a sinusoidal sequence:

$$u(k) = \sin(ωkh)$$

then after transients have died out, the output will also be sinusoidal with the same frequency but with different amplitude and phase:

$$y(k) = |H(e^{iωh})|\sin(ωkh + \arg H(e^{iωh}))$$

The frequency response is thus:

$$H(e^{iωh}) = |H(e^{iωh})|e^{i\arg H(e^{iωh})}$$

This is obtained by substituting z = e^{iωh} in the pulse-transfer function H(z).

For a continuous-time system, the input u(t) = sin(ωt) gives the output:

$$y(t) = |G(iω)|\sin(ωt + \arg G(iω))$$

The key difference is that for the sampled-data system, the frequency response is periodic in ω with period ωₛ = 2π/h (the sampling frequency). This is due to aliasing.

**Example:**

Consider the system:

$$H(z) = \frac{0.368}{z - 0.368}$$

which is obtained by zero-order hold sampling of G(s) = 1/(s+1) with h = 1. The frequency response is:

$$H(e^{iω}) = \frac{0.368}{e^{iω} - 0.368}$$

Figure 8 shows the Bode plot of this frequency response. Notice the periodicity with period 2π.

### Lifting

For a sampled-data system, the continuous-time output between sampling instants depends on when the sinusoidal input is applied relative to the sampling instants. This time-varying behavior can be analyzed using the concept of lifting.

The idea is to consider the system as having an infinite number of inputs and outputs, one for each point in the sampling period. This transforms the time-varying sampled-data system into a time-invariant infinite-dimensional system.

For a sinusoidal input u(t) = sin(ωt + φ), the continuous-time output at time kh + τ, where 0 ≤ τ < h, can be written as:

$$y(kh + τ) = |H(e^{iωh}, τ)|\sin(ωkh + ωτ + φ + \arg H(e^{iωh}, τ))$$

where H(e^{iωh}, τ) is the lifted frequency response that depends on both ω and τ.

### Practical Consequences

The periodic nature of the frequency response has several practical consequences:

1. **Aliasing:** Frequencies ω and ω + nωₛ give the same sampled sequence and thus have the same effect on the sampled output.

2. **Folding frequency:** The Nyquist frequency ω_N = ωₛ/2 is the highest frequency that can be uniquely represented in a sampled-data system.

3. **Bode plots:** When plotting frequency response for sampled-data systems, it is sufficient to consider frequencies up to the Nyquist frequency.

4. **Intersample behavior:** The continuous-time output between samples can have frequency content above the Nyquist frequency even if the sampled output does not.

5. **Design implications:** When designing controllers, it is important to consider both the sampled frequency response and the continuous-time intersample behavior.

### Summary

Key points about frequency response in sampled-data systems:

- The frequency response H(e^{iωh}) is periodic with period ωₛ
- Aliasing causes frequencies above the Nyquist frequency to fold back
- The intersample behavior can be analyzed using lifting
- Both sampled and continuous-time frequency responses are important for controller design

---

## 5. Control Design and Specifications

This chapter discusses the control design problem for sampled-data systems. The design involves selecting controller parameters to meet specifications on closed-loop performance.

### The Process

The process or plant to be controlled is assumed to be described by a linear time-invariant model, either in state-space form:

$$\frac{dx}{dt} = Ax + Bu$$
$$y = Cx + Du$$

or as a transfer function G(s). The sampled-data model is obtained using the methods in Chapter 3.

### Admissible Controls

Not all control signals are admissible in a sampled-data system. The constraints typically include:

- **Magnitude constraints:** |u(t)| ≤ u_max
- **Rate constraints:** |du/dt| ≤ r_max
- **Sampling constraints:** u(t) is piecewise constant with period h

These constraints must be considered in the controller design.

### Design Parameters

The controller typically has several design parameters that can be adjusted to meet specifications. Examples include:

- Feedback gains
- Observer/estimator gains
- Weighting matrices in optimal control
- Pole locations
- Sampling period h

### Criteria and Specifications

Common specifications for control systems include:

**Time-domain specifications:**
- Rise time
- Settling time
- Overshoot
- Steady-state error

**Frequency-domain specifications:**
- Bandwidth
- Phase margin
- Gain margin
- Sensitivity function bounds

**Optimal control specifications:**
- Minimize quadratic cost function
- H∞ norm bounds
- H₂ optimal control

### Robustness

Robustness refers to the ability of the control system to maintain performance when there are uncertainties in the process model. Important robustness measures include:

- **Gain margin:** How much the loop gain can vary before instability
- **Phase margin:** How much phase lag can be added before instability
- **Sensitivity:** How sensitive the closed-loop system is to parameter variations

### Attenuation of Load Disturbances

Load disturbances are signals that affect the process but are not measured. The controller should minimize the effect of load disturbances on the output. This is typically achieved by:

- High loop gain at frequencies where disturbances occur
- Integral action in the controller
- Feedforward compensation if disturbance is measured

The sensitivity function S = 1/(1 + PC) characterizes load disturbance attenuation, where P is the process and C is the controller.

### Injection of Measurement Noise

Measurement noise is inevitable in any real system. The controller should not amplify measurement noise excessively. This is typically achieved by:

- Low controller gain at high frequencies
- Filtering of measurements
- Low-pass characteristics in the loop transfer function

The complementary sensitivity function T = PC/(1 + PC) characterizes measurement noise injection.

### Command Signals Following

The controller should make the output follow command signals with acceptable performance. This can be specified as:

- Rise time and overshoot for step commands
- Tracking error for ramp commands
- Bandwidth for general command signals

Two-degrees-of-freedom controllers can provide better command following performance than simple feedback controllers.

### Summary

Control design involves:

- Selecting a control structure (feedback, feedforward, two-degrees-of-freedom)
- Choosing design parameters to meet specifications
- Balancing conflicting requirements (performance vs robustness, command following vs disturbance rejection)
- Considering constraints on control signals
- Ensuring adequate robustness

The next chapters will discuss specific design methods for sampled-data controllers.

---

## 6. Approximation of Analog Controllers

A simple approach to computer control is to design a continuous-time controller and then approximate it with a discrete-time controller. This chapter discusses several approximation methods.

### State Model of the Controller

If the continuous-time controller is given in state-space form:

$$\frac{dx_c}{dt} = A_cx_c + B_cy$$
$$u = C_cx_c + D_cy$$

then the discrete-time approximation can be obtained by sampling this system with sampling period h:

$$x_c(k+1) = Φ_cx_c(k) + Γ_cy(k)$$
$$u(k) = C_cx_c(k) + D_cy(k)$$

where:

$$Φ_c = e^{A_ch}$$
$$Γ_c = \int_0^h e^{A_cτ}dτ \cdot B_c$$

This gives an exact discrete-time equivalent of the continuous-time controller at the sampling instants.

### Transfer Functions

If the continuous-time controller is given as a transfer function:

$$C(s) = \frac{U(s)}{Y(s)}$$

then several approximation methods can be used to obtain a discrete-time controller C(z).

**Backward difference (Euler's method):**

$$s ≈ \frac{1-z^{-1}}{h}$$

**Forward difference:**

$$s ≈ \frac{z-1}{h}$$

**Tustin's approximation (Bilinear transform):**

$$s ≈ \frac{2}{h}\frac{z-1}{z+1}$$

**Example:**

For the continuous-time controller:

$$C(s) = K\frac{s+a}{s+b}$$

Tustin's approximation gives:

$$C(z) = K\frac{\frac{2}{h}\frac{z-1}{z+1}+a}{\frac{2}{h}\frac{z-1}{z+1}+b} = K\frac{(2/h+a)z-(2/h-a)}{(2/h+b)z-(2/h-b)}$$

### Frequency Prewarping

A problem with Tustin's approximation is that the frequency response of the discrete-time controller does not match the continuous-time controller exactly. This can be corrected by frequency prewarping.

The idea is to modify Tustin's approximation so that the frequency responses match at a specific frequency ω₀:

$$s ≈ \frac{ω_0}{\tan(ω_0h/2)}\frac{z-1}{z+1}$$

This ensures that C(e^{iω₀h}) = C(iω₀).

### Step Invariance

The step invariance method ensures that the step response of the discrete-time controller matches the continuous-time controller at the sampling instants.

For a controller:

$$C(s) = \frac{N(s)}{D(s)}$$

the step-invariant discrete-time controller is:

$$C(z) = (1-z^{-1})Z\left[\frac{N(s)}{sD(s)}\right]$$

### Ramp Invariance

Similar to step invariance, ramp invariance ensures that the ramp response matches at sampling instants:

$$C(z) = \frac{(1-z^{-1})^2}{hz^{-1}}Z\left[\frac{N(s)}{s^2D(s)}\right]$$

### Comparison of Approximations

The different approximation methods have different characteristics:

**Backward difference:**
- Maps the left half s-plane inside the unit circle
- Always gives a stable discrete-time controller if continuous-time controller is stable
- Poor frequency response approximation

**Forward difference:**
- Can give unstable discrete-time controller even if continuous-time controller is stable
- Not recommended

**Tustin's approximation:**
- Good frequency response approximation
- Maps imaginary axis to unit circle
- Stable continuous-time controller gives stable discrete-time controller
- Most commonly used method

**Frequency prewarping:**
- Exact frequency response match at one frequency
- Good approximation over a wider frequency range
- Recommended when matching at a specific frequency is important

**Step/ramp invariance:**
- Exact match of time response at sampling instants
- Good for slow systems
- Can give poor frequency response approximation

### Selection of Sampling Interval and Antialiasing Filters

The sampling period h and antialiasing filter must be chosen together. Guidelines:

1. **Shannon's theorem:** h ≤ π/ω_max where ω_max is the highest significant frequency

2. **Practical rule:** ω_s = 2π/h should be 20–30 times the closed-loop bandwidth

3. **Antialiasing filter:** Cut-off frequency should be 0.1–0.5 times the sampling frequency

4. **Trade-offs:**
   - Shorter h gives better approximation but requires more computation
   - Longer h is more economical but requires better controller design
   - Antialiasing filter introduces phase lag that must be considered in design

### Summary

Approximation of continuous-time controllers is a simple approach to computer control:

- Several approximation methods available
- Tustin's approximation (bilinear transform) most commonly used
- Sampling period should be chosen to give good approximation
- Antialiasing filters needed to prevent aliasing
- For best performance, direct discrete-time design methods should be used

---

## 7. Feedforward Design

Feedforward control can significantly improve control performance when disturbances or command signals are measurable. This chapter discusses feedforward design for sampled-data systems.

### Reduction of Measurable Disturbances by Feedforward

Consider a system where a measurable disturbance v affects the process. The control signal can be decomposed as:

$$u = u_{fb} + u_{ff}$$

where u_fb is the feedback part and u_ff is the feedforward part.

The feedforward compensator should be designed so that the effect of the disturbance on the output is canceled. In the ideal case:

$$G_{yv}(s) + G_{yu}(s)F(s) = 0$$

where G_yv is the transfer function from disturbance to output, G_yu is the transfer function from control to output, and F is the feedforward compensator.

This gives:

$$F(s) = -\frac{G_{yv}(s)}{G_{yu}(s)}$$

For the discrete-time case:

$$F(z) = -\frac{H_{yv}(z)}{H_{yu}(z)}$$

### System Inverses

The feedforward compensator often requires inverting the process transfer function. This is not always possible or desirable:

**Causality:** If G_yu has more poles than zeros (relative degree > 0), then 1/G_yu is non-causal and cannot be implemented.

**Stability:** If G_yu has zeros in the right half-plane (non-minimum phase), then 1/G_yu is unstable.

**Solutions:**

1. **Approximation:** Use a causal stable approximation of the inverse

2. **Delay:** Introduce a delay in the feedforward path:
   $$F(s) = -e^{-Ts}\frac{G_{yv}(s)}{G_{yu}(s)}$$

3. **Partial cancellation:** Only cancel the stable, minimum-phase part

For discrete-time systems, the same issues arise:

- Zeros outside the unit circle (unstable zeros) should not be inverted
- Relative degree must be handled by introducing delays (z^{-d} factors)

**Example:**

For a process:

$$H(z) = \frac{b_0 + b_1z^{-1}}{1 + a_1z^{-1}}$$

with b₀ ≠ 0, the inverse is:

$$H^{-1}(z) = \frac{1 + a_1z^{-1}}{b_0 + b_1z^{-1}}$$

This is causal. If b₀ = 0, the system has a delay and the inverse would be non-causal.

### Using Feedforward to Improve Response to Command Signals

Feedforward can also be used to improve the response to command signals. In a two-degrees-of-freedom controller:

$$u = F_f(z)u_c - F_b(z)y$$

the feedforward part F_f can be designed to give desired command following while the feedback part F_b provides disturbance rejection and robustness.

For perfect command following:

$$F_f(z) = \frac{1}{H(z)}$$

but this may not be realizable. A practical approach is to design F_f to give a desired closed-loop transfer function:

$$\frac{Y(z)}{U_c(z)} = H_d(z)$$

This requires:

$$F_f(z) = \frac{H_d(z)}{H(z)[1 + F_b(z)H(z)]}$$

A common choice is to make H_d(z) a low-pass filter with appropriate bandwidth.

### Summary

Feedforward control provides significant benefits:

- Measurable disturbances can be canceled before affecting output
- Command following can be improved
- Feedback requirements are reduced

Key considerations:

- Feedforward requires accurate process model
- Inverse may be non-causal or unstable
- Delays and approximations may be needed
- Combine with feedback for robustness

---

## 8. PID Control

PID (Proportional-Integral-Derivative) control is the most common control algorithm in industry. This chapter discusses implementation of PID control in sampled-data systems.

### Modification of Linear Response

The basic PID controller in continuous time is:

$$u(t) = K\left[e(t) + \frac{1}{T_i}\int_0^t e(τ)dτ + T_d\frac{de(t)}{dt}\right]$$

where e = u_c - y is the control error, K is the proportional gain, T_i is the integral time, and T_d is the derivative time.

In practice, the derivative term is modified to reduce noise sensitivity:

$$u(t) = K\left[e(t) + \frac{1}{T_i}\int_0^t e(τ)dτ + \frac{T_d}{1 + T_ds}De(t)\right]$$

where D(·) denotes the filtered derivative.

### Discretization

The PID controller can be discretized using the methods in Chapter 6. A common approach is:

**Integral term:** Use trapezoidal approximation (Tustin):
$$\int_0^t e(τ)dτ ≈ \frac{h}{2}\sum_{k=0}^{n}[e(k) + e(k-1)]$$

**Derivative term:** Use backward difference:
$$\frac{de}{dt} ≈ \frac{e(k) - e(k-1)}{h}$$

This gives the discrete-time PID controller:

$$u(k) = K\left[e(k) + \frac{h}{T_i}\sum_{j=0}^{k}e(j) + \frac{T_d}{h}(e(k) - e(k-1))\right]$$

Different discretization methods lead to different coefficient values. Table 3 compares three common approximation methods for the PID controller in the general form:

$$R(q)u(kh) = T(q)u_c(kh) - S(q)y(kh)$$

where $$R(q) = (q-1)(q-a_d)$$, $$S(q) = s_0 + s_1q^{-1} + s_2q^{-2}$$, and $$T(q) = t_0 + t_1q^{-1} + t_2q^{-2}$$.

**Table 3: Coefficients for Different PID Discretization Methods**

| Coefficient | Special | Tustin | Ramp Equivalence |
|------------|---------|--------|------------------|
| $$a_d$$ | $$\frac{T_d}{Nh+T_d}$$ | $$\frac{2T_d-Nh}{2T_d+Nh}$$ | $$\exp(-Nh/T_d)$$ |
| $$b_d$$ | $$\frac{Na_d}{h}$$ | $$\frac{2NT_d}{2T_d+Nh}$$ | $$\frac{N(1-a_d)}{h}$$ |
| $$b_i$$ | $$\frac{h}{T_i}$$ | $$\frac{h}{2T_i}$$ | $$\frac{h}{2T_i}$$ |
| $$s_0$$ | $$K(1+b_d)$$ | $$K(1+b_i+b_d)$$ | $$K(1+b_i+b_d)$$ |
| $$s_1$$ | $$-K(1+a_d+2b_d-b_i)$$ | $$-K[1+a_d+2b_d-b_i(1-a_d)]$$ | $$-K[1+a_d+2b_d-b_i(1-a_d)]$$ |
| $$s_2$$ | $$K(a_d+b_d-b_ia_d)$$ | $$K(a_d+b_d-b_ia_d)$$ | $$K(a_d+b_d-b_ia_d)$$ |
| $$t_0$$ | $$Kb$$ | $$K(b+b_i)$$ | $$K(b+b_i)$$ |
| $$t_1$$ | $$-K[b(1+a_d)-b_i]$$ | $$-K[b(1+a_d)-b_i(1-a_d)]$$ | $$-K[b(1+a_d)-b_i(1-a_d)]$$ |
| $$t_2$$ | $$Ka_d(b-b_i)$$ | $$Ka_d(b-b_i)$$ | $$Ka_d(b-b_i)$$ |

where $$b$$ is the setpoint weighting factor, $$K$$ is the proportional gain, $$T_i$$ is the integral time, $$T_d$$ is the derivative time, $$N$$ is the derivative filter constant, and $$h$$ is the sampling period.

### Incremental Algorithms

The standard form above is called the position form. An alternative is the velocity or incremental form:

$$\Delta u(k) = u(k) - u(k-1) = K\left[e(k) - e(k-1) + \frac{h}{T_i}e(k) + \frac{T_d}{h}(e(k) - 2e(k-1) + e(k-2))\right]$$

The incremental form has several advantages:

- Does not require storage of integral sum
- Less sensitive to integration drift
- Easier to handle manual/automatic mode switching
- Better for bumpless transfer

### Integrator Windup

Integrator windup occurs when the control signal saturates. The integral term continues to integrate the error, leading to large control signals and poor performance when the saturation is removed.

**Anti-windup schemes:**

1. **Conditional integration:** Stop integration when control signal saturates
   ```
   if u_min < u < u_max:
       I = I + (h/T_i)e
   ```

2. **Back-calculation:** Feed back the difference between actual and computed control signal
   ```
   I = I + (h/T_i)e + (h/T_t)(u_actual - u_computed)
   ```
   where T_t is the tracking time constant

3. **Incremental form:** Naturally provides some anti-windup protection

### Operational Aspects

Practical PID implementation includes several features:

**Set-point weighting:**
$$u = K\left[\beta u_c - y + \frac{1}{T_i}\int(u_c - y)dτ + T_d\frac{d}{dt}(\gamma u_c - y)\right]$$

where β and γ are weighting factors (typically β = 1, γ = 0 to reduce overshoot).

**Filtering:**
- Measurement filtering to reduce noise
- Derivative filtering to reduce noise amplification

**Bumpless transfer:**
- Smooth transition between manual and automatic mode
- Initialize integral term appropriately

**Mode handling:**
- Manual mode
- Automatic mode
- Tracking mode (following external signal)

### Computer Code

**Python implementation of digital PID controller with anti-windup using ctrlsys:**

```python
#!/usr/bin/env python3
"""Digital PID Controller with anti-windup (converted from Java Listing 1)"""
import numpy as np
from ctrlsys import ab04md

def tf_to_ss(num, den):
    """Convert SISO transfer function to controllable canonical state-space."""
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    den = den / den[0]
    num = num / den[0]
    n = len(den) - 1
    if n == 0:
        return (np.zeros((0, 0), order='F'), np.zeros((0, 1), order='F'),
                np.zeros((1, 0), order='F'), np.array([[num[0]]], order='F'))
    num_padded = np.zeros(n + 1)
    num_padded[n + 1 - len(num):] = num
    A = np.zeros((n, n), order='F', dtype=float)
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]
    B = np.zeros((n, 1), order='F', dtype=float)
    B[-1, 0] = 1.0
    C = np.zeros((1, n), order='F', dtype=float)
    d0 = num_padded[0]
    for i in range(n):
        C[0, n - 1 - i] = num_padded[i + 1] - d0 * den[i + 1]
    D = np.array([[d0]], order='F', dtype=float)
    return A, B, C, D

class PIDController:
    """PID controller with anti-windup and setpoint weighting."""

    def __init__(self, K=4.4, Ti=0.4, Td=0.2, h=0.03, Tt=10.0,
                 N=10.0, b=1.0, ulow=-1.0, uhigh=1.0):
        self.params = {'K': K, 'Ti': Ti, 'Td': Td, 'h': h,
                      'Tt': Tt, 'N': N, 'b': b, 'ulow': ulow, 'uhigh': uhigh}
        self.params['bi'] = K * h / Ti
        self.params['ar'] = h / Tt
        self.params['ad'] = Td / (Td + N * h)
        self.params['bd'] = K * N * self.params['ad']
        self.states = {'I': 0.0, 'D': 0.0, 'yold': 0.0}
        self.signals = {'uc': 0.0, 'y': 0.0, 'v': 0.0, 'u': 0.0}

    def calculate_output(self, uc, y):
        """Calculate PID output u(k) given setpoint uc and measurement y."""
        p = self.params
        self.signals['uc'], self.signals['y'] = uc, y
        P = p['K'] * (p['b'] * uc - y)
        self.states['D'] = p['ad'] * self.states['D'] - p['bd'] * (y - self.states['yold'])
        self.signals['v'] = P + self.states['I'] + self.states['D']
        self.signals['u'] = np.clip(self.signals['v'], p['ulow'], p['uhigh'])
        return self.signals['u']

    def update_state(self, u):
        """Update states after output applied (anti-windup)."""
        p = self.params
        self.states['I'] += p['bi'] * (self.signals['uc'] - self.signals['y']) + \
                           p['ar'] * (u - self.signals['v'])
        self.states['yold'] = self.signals['y']

# Example usage
if __name__ == "__main__":
    # Create plant: G(s) = 1/(s^2 + 2s + 1)
    A, B, C, D = tf_to_ss([1], [1, 2, 1])
    dt = 0.01

    # Discretize using Tustin
    A_d, B_d, C_d, D_d, _ = ab04md('C',
        np.asfortranarray(A), np.asfortranarray(B),
        np.asfortranarray(C), np.asfortranarray(D),
        alpha=1.0, beta=2.0/dt)

    # Create PID controller
    pid = PIDController(K=3.0, Ti=1.0, Td=0.2, h=dt)

    # Simulation loop
    setpoint = 1.0
    x = np.zeros(A_d.shape[0])

    for k in range(1000):
        y = (C_d @ x).item()
        u = pid.calculate_output(setpoint, y)
        pid.update_state(u)
        x = A_d @ x + B_d.flatten() * u

        if k % 100 == 0:
            print(f"t={k*dt:.2f}s: y={y:.3f}, u={u:.3f}")
```

**Key features of this implementation:**

1. **Class-based design** for easy parameter management
2. **Anti-windup** using back-calculation method: `I += ar*(u_actual - u_computed)`
3. **Setpoint weighting** (parameter b) to reduce overshoot
4. **Filtered derivative** to reduce noise sensitivity
5. **ctrlsys integration** for plant simulation and control system design

The complete implementation is available in `pid_controller.py` with additional features including bumpless parameter changes and comprehensive simulation capabilities.

### Tuning

PID controller tuning methods:

**Ziegler-Nichols methods:**
- Step response method
- Ultimate gain method

**Relay auto-tuning:**
- Uses relay feedback to find critical point
- Automatically computes PID parameters

**Lambda tuning:**
- Specifies desired closed-loop time constant
- Provides smooth response

**Optimization-based:**
- Minimize performance index (ISE, IAE, ITAE)
- Can be done offline or online

For sampled-data PID control, the sampling period should be chosen as:
$$h ≈ 0.1T_d \text{ to } 0.5T_d$$

where T_d is the dominant time constant of the process.

### Summary

PID control in sampled-data systems:

- Discretization of continuous-time PID using approximations
- Incremental form often preferred for practical reasons
- Anti-windup essential for good performance
- Many practical considerations (filtering, mode handling, etc.)
- Tuning methods available for automatic parameter selection
- Sampling period should be sufficiently short

PID control is simple and effective for many applications, but advanced methods may be needed for complex systems.

---

## 9. Pole-placement Design

Pole-placement is a design method where the controller parameters are chosen to place the closed-loop poles at desired locations. This chapter discusses pole-placement design for sampled-data systems.

The pole-placement problem can be stated as follows: Given a process with pulse-transfer function:

$$H(z) = \frac{B(z)}{A(z)}$$

and desired closed-loop characteristic polynomial P(z), find a controller with transfer function:

$$C(z) = \frac{S(z)}{R(z)}$$

such that the closed-loop system has characteristic polynomial P(z).

The closed-loop characteristic polynomial is:

$$A(z)R(z) + B(z)S(z) = P(z)$$

This is called the Diophantine equation. Solving this equation for R(z) and S(z) gives the controller parameters.

**Example:**

Consider a process:

$$H(z) = \frac{b_0z + b_1}{z^2 + a_1z + a_2}$$

and desired characteristic polynomial:

$$P(z) = z^3 + p_1z^2 + p_2z + p_3$$

The controller is assumed to have the form:

$$C(z) = \frac{s_0z + s_1}{z^2 + r_1z + r_2}$$

The Diophantine equation becomes:

$$(z^2 + a_1z + a_2)(z^2 + r_1z + r_2) + (b_0z + b_1)(s_0z + s_1) = z^3 + p_1z^2 + p_2z + p_3$$

Expanding and equating coefficients gives a system of linear equations that can be solved for r₁, r₂, s₀, s₁.

### The Diophantine Equation

The general Diophantine equation:

$$A(z)R(z) + B(z)S(z) = P(z)$$

where:
- A(z) has degree n
- B(z) has degree m ≤ n
- R(z) has degree n_r
- S(z) has degree n_s

For the equation to have a unique solution, we need:

$$\text{deg}(P) = n + n_r + 1 = m + n_s + 1$$

This gives:

$$n_r = m$$
$$n_s = n - 1$$
$$\text{deg}(P) = n + m + 1$$

### Causality Conditions

For the controller to be causal (implementable), we need:

$$\text{deg}(S) ≤ \text{deg}(R)$$

This is automatically satisfied if n_s = n - 1 and n_r = m ≥ 0.

### Summary of the Pole-placement Design Procedure

1. **Model the process:** Obtain pulse-transfer function H(z) = B(z)/A(z)

2. **Choose desired poles:** Select closed-loop characteristic polynomial P(z)
   - Poles should be inside unit circle for stability
   - Pole locations determine speed and damping
   - Typical choices: Bessel, Butterworth patterns

3. **Set up Diophantine equation:** AR + BS = P

4. **Solve for controller parameters:** Find R(z) and S(z)
   - Equate coefficients
   - Solve linear system of equations
   - Check for numerical conditioning

5. **Implement controller:**
   $$u(k) = \frac{S(z)}{R(z)}e(k)$$
   where e = u_c - y

6. **Verify performance:** Simulate or test

**Example:**

For a double integrator:

$$H(z) = \frac{0.005z + 0.005}{z^2 - 2z + 1}$$

Choose desired poles at z = 0.5 ± 0.5i (giving ζ ≈ 0.7):

$$P(z) = (z - 0.5 - 0.5i)(z - 0.5 + 0.5i)(z - 0.5) = z^3 - 1.5z^2 + 1z - 0.25$$

Solving the Diophantine equation gives the controller.

### Introduction of Integrators

To achieve zero steady-state error for step commands, an integrator can be included in the controller. This is done by factoring:

$$R(z) = (z - 1)R'(z)$$

The Diophantine equation becomes:

$$A(z)(z - 1)R'(z) + B(z)S(z) = P(z)$$

For step disturbances at the output, the factor (z - 1) should be in A(z) or R(z).

For step disturbances at the input, the factor (z - 1) should appear in both the original A(z) (describing the disturbance model) and can be absorbed in the design.

### Summary

Pole-placement design:

- Direct method for choosing closed-loop dynamics
- Solution via Diophantine equation
- Can include integral action for zero steady-state error
- Can handle constraints and additional specifications
- Straightforward computation
- May give controllers with high gain sensitivity

The method is intuitive and provides good control but requires:
- Good process model
- Appropriate choice of desired poles
- Consideration of robustness

---

## 10. Optimization Based Design

This chapter discusses design methods based on optimization of performance criteria. Two main approaches are covered: Linear Quadratic (LQ) design and H∞ design.

### Linear Quadratic (LQ) Design

The LQ optimal control problem is to find a control signal that minimizes a quadratic cost function:

$$J = \sum_{k=0}^{\infty}[x^T(k)Q_1x(k) + u^T(k)Q_2u(k)]$$

subject to the system dynamics:

$$x(k+1) = Φx(k) + Γu(k)$$

where Q₁ ≥ 0 and Q₂ > 0 are weighting matrices.

The solution is a state feedback controller:

$$u(k) = -Lx(k)$$

where the feedback gain L is given by:

$$L = (Γ^TKΓ + Q_2)^{-1}Γ^TKΦ$$

and K is the solution to the discrete-time Riccati equation:

$$K = Φ^TKΦ - Φ^TKΓ(Γ^TKΓ + Q_2)^{-1}Γ^TKΦ + Q_1$$

The Riccati equation can be solved iteratively or using efficient numerical algorithms.

**Properties of LQ control:**
- Optimal with respect to the cost function
- Guaranteed stability margins (gain margin ≥ 0.5, phase margin ≥ 60°)
- Systematic design procedure
- State feedback requires full state measurement or observer

### How to Find the Weighting Matrices?

The choice of weighting matrices Q₁ and Q₂ determines the trade-off between state regulation and control effort. Guidelines:

1. **Output weighting:**
   $$Q_1 = C^TQC$$
   where Q penalizes output errors

2. **Bryson's rule:**
   $$Q_1(i,i) = \frac{1}{\text{maximum acceptable value of } x_i^2}$$
   $$Q_2(j,j) = \frac{1}{\text{maximum acceptable value of } u_j^2}$$

3. **Trial and error:**
   - Start with Q₁ = I, Q₂ = ρI
   - Vary ρ to adjust control effort
   - Increase Q₁(i,i) to tighten control of state i

4. **Inverse optimal control:**
   - Start with desired closed-loop poles
   - Find Q₁, Q₂ that give those poles

**Example:**

For the disk drive system:

$$x(k+1) = \begin{bmatrix}1 & h \\ 0 & 1\end{bmatrix}x(k) + \begin{bmatrix}h^2/2 \\ h\end{bmatrix}u(k)$$

$$y(k) = \begin{bmatrix}1 & 0\end{bmatrix}x(k)$$

Choose Q₁ = diag(1, 0) to penalize position error, Q₂ = ρ to limit control effort. Solving the Riccati equation gives the optimal feedback gain.

### Kalman Filters and LQG Control

When the state is not measurable, an observer or Kalman filter must be used. The Kalman filter is the optimal state estimator for systems with Gaussian noise.

Consider the system:

$$x(k+1) = Φx(k) + Γu(k) + w(k)$$
$$y(k) = Cx(k) + v(k)$$

where w is process noise with covariance R₁ and v is measurement noise with covariance R₂.

The Kalman filter is:

$$\hat{x}(k+1|k) = Φ\hat{x}(k|k-1) + Γu(k) + K_f[y(k) - C\hat{x}(k|k-1)]$$

where the filter gain K_f is:

$$K_f = ΦPC^T(CPC^T + R_2)^{-1}$$

and P is the solution to the discrete-time Riccati equation:

$$P = ΦPΦ^T - ΦPC^T(CPC^T + R_2)^{-1}CPΦ^T + R_1$$

**LQG control:** Combining LQ controller with Kalman filter gives LQG (Linear Quadratic Gaussian) control:

$$u(k) = -L\hat{x}(k)$$

where ̂x is the Kalman filter estimate.

**Separation principle:** The controller and estimator can be designed independently. The closed-loop poles are the union of controller poles and estimator poles.

**Properties:**
- Optimal for quadratic cost with Gaussian noise
- May have poor robustness properties
- Loop transfer recovery techniques can improve robustness

### H∞ Design

H∞ design formulates the control problem as minimizing the worst-case effect of disturbances on outputs. The goal is to find a controller that makes the H∞ norm of a transfer function small.

The H∞ norm is defined as:

$$||G||_∞ = \sup_{ω}|G(e^{iω})|$$

This represents the maximum gain over all frequencies.

**Standard H∞ problem:**

Given a generalized plant:

$$\begin{bmatrix}z \\ y\end{bmatrix} = G\begin{bmatrix}w \\ u\end{bmatrix}$$

where:
- w is exogenous input (disturbances, noise, commands)
- u is control input
- z is regulated output (tracking error, control effort)
- y is measured output

Find controller K such that:

$$||F_l(G, K)||_∞ < γ$$

where F_l(G, K) is the lower linear fractional transformation.

**Solution methods:**
- Riccati equation approach
- Linear matrix inequalities (LMI)
- Descriptor system formulation

**Weighting functions:**

To shape the closed-loop response, weighting functions are introduced:

$$z = \begin{bmatrix}W_1(z)e \\ W_2(z)u\end{bmatrix}$$

where:
- W₁ specifies desired error attenuation
- W₂ limits control effort

**Example:**

For disturbance attenuation, choose:

$$W_1(z) = \frac{z - a}{z - 1}$$

to emphasize low-frequency disturbance rejection.

**Properties of H∞ control:**
- Handles worst-case disturbances
- Allows shaping of frequency response
- Guaranteed robustness to specific uncertainties
- Can be conservative

### Summary

Optimization-based design methods:

**LQ/LQG:**
- Minimize quadratic cost
- Optimal for Gaussian noise
- Systematic procedure
- May need robustness enhancement

**H∞:**
- Minimize worst-case gain
- Direct robustness specifications
- Frequency shaping via weights
- Can be conservative

Both methods require:
- Good process model
- Appropriate choice of weights
- Numerical solution of Riccati equations or LMIs

These methods are more advanced than pole-placement but can provide better performance and robustness for complex systems.

---

## 11. Practical Issues

This chapter discusses practical issues that arise when implementing controllers on digital computers.

### Controller Implementation and Computational Delay

The control algorithm requires time to execute. This introduces a delay between measurement and actuation. The timing can be characterized by:

- **Input-output delay:** Time from reading sensor to writing actuator
- **Execution time:** Time to execute control algorithm
- **Latency:** Time from event to response

**Effects of computational delay:**
- Phase lag in loop transfer function
- Reduced phase margin
- Potential instability

**Handling computational delay:**

1. **Fast enough computer:** Make execution time << h

2. **Compensation:** Include delay in model:
   $$H_d(z) = z^{-1}H(z)$$

3. **Prediction:** Predict future state and control for time k+1

4. **Fixed delay:** If delay is constant, compensate in design

**Example:**

For a system with one sample delay:

$$y(k) = H(z)u(k-1)$$

The control law becomes:

$$u(k) = C(z)e(k-1)$$

This introduces an additional z⁻¹ factor in the loop transfer function.

### Controller Representation and Numerical Roundoff

Finite word length in computers causes numerical errors. The controller can be implemented in different forms with different numerical properties.

**Direct form:**

$$u(k) = \frac{1}{r_0}[s_0e(k) + s_1e(k-1) + ... - r_1u(k-1) - ...]$$

**Cascade form:**

$$C(z) = K\prod_{i=1}^{n}\frac{z - z_i}{z - p_i}$$

Implement as cascade of first and second-order sections.

**Parallel form:**

$$C(z) = K + \sum_{i=1}^{n}\frac{c_i}{z - p_i}$$

**State-space form:**

$$x_c(k+1) = Φ_cx_c(k) + Γ_ce(k)$$
$$u(k) = C_cx_c(k) + D_ce(k)$$

**Comparison:**

- **Direct form:** Simple, but sensitive to roundoff for high-order controllers
- **Cascade form:** Better numerical properties, easy to implement
- **Parallel form:** Good for partial fraction expansions
- **State-space:** Best numerical properties for high-order controllers

**Fixed-point vs floating-point:**

- **Fixed-point:** Limited range and precision, but faster
- **Floating-point:** Wide range, good precision, slower
- Modern processors have fast floating-point

**Recommendations:**
- Use floating-point if available
- Use cascade or state-space form for high-order controllers
- Scale signals to use full range
- Check for overflow/underflow

### A-D and D-A Quantization

A-D and D-A converters have finite resolution, causing quantization errors.

**A-D quantization:**

The quantization error is bounded by:

$$|e_q| ≤ \frac{q}{2}$$

where q is the quantization level. For an n-bit converter with range [-V, V]:

$$q = \frac{2V}{2^n}$$

**Effects:**
- Adds noise to measurement
- Equivalent to measurement noise
- Can cause limit cycles

**D-A quantization:**

Similarly causes quantization of control signal.

**Limit cycles:**

In systems with high gain and quantization, limit cycles can occur:

$$x(k+1) = ax(k) + bu(k)$$
$$u(k) = -Q[Lx(k)]$$

where Q is quantization. For |a - bL| < 1, the system may oscillate with small amplitude.

**Prevention:**
- Use sufficient resolution (12-16 bits usually adequate)
- Add dither (small random noise)
- Design controller with robustness to quantization

### Sampling Period Selection

The sampling period h is a crucial design parameter. Guidelines:

**Shannon's theorem:**
$$h ≤ \frac{π}{ω_{max}}$$

**Based on bandwidth:**
$$ω_sh ≥ 20 \text{ to } 30$$
where ω_s is closed-loop bandwidth

**Based on rise time:**
$$h ≤ \frac{T_r}{10}$$

**Based on phase margin:**
$$h ≤ \frac{φ_m}{6ω_c}$$
where φ_m is desired phase margin and ω_c is crossover frequency

**Trade-offs:**
- **Short h:** Better performance, more computation, more noise
- **Long h:** Less computation, worse performance, may be unstable

**Multi-rate systems:**

Different loops can have different sampling periods:
- Fast loops: h_fast for inner loops, fast dynamics
- Slow loops: h_slow = Nh_fast for outer loops, slow dynamics

### Saturations and Windup

Actuators have physical limitations causing saturation. This is a nonlinearity that can cause:

**Integrator windup:**
- Integral term grows large during saturation
- Large overshoot when saturation ends
- Poor performance

**Other effects:**
- Instability
- Limit cycles
- Reduced performance

**Anti-windup techniques:**

1. **Conditional integration:**
   ```
   if |u| < u_max:
       integrate error
   else:
       don't integrate
   ```

2. **Back-calculation:**
   ```
   I = I + (h/T_i)e + (h/T_t)(u_sat - u_comp)
   ```

3. **Observer with saturation:**
   Include saturation in observer model

4. **Model predictive control:**
   Explicitly handle constraints in optimization

**Example - Back-calculation:**

For a PI controller:

$$u = Ke + KI$$
$$I(k+1) = I(k) + \frac{h}{T_i}e(k)$$

With saturation:

$$u_{sat} = sat(u, u_{min}, u_{max})$$

Add anti-windup:

$$I(k+1) = I(k) + \frac{h}{T_i}e(k) + \frac{h}{T_t}(u_{sat} - u)$$

where T_t is tracking time constant (typically T_t = T_i).

**Bumpless transfer:**

When switching between modes (manual/automatic), initialize controller state to avoid bumps:

```
// When switching to automatic
if switching_to_auto:
    I = u_manual - P  // Initialize integral term
    u = u_manual       // Match manual output
```

### Summary

Practical implementation requires attention to:

- **Computational delay:** Can destabilize, must be compensated
- **Numerical accuracy:** Use appropriate representation
- **Quantization:** Usually not a problem with 12+ bit converters
- **Sampling period:** Multiple guidelines, trade-offs
- **Saturation:** Must use anti-windup techniques

These issues can significantly affect performance and must be considered in any practical implementation.

---

## 12. Real-time Implementation

Computer control systems must execute in real-time with precise timing. This chapter discusses real-time programming and implementation issues.

### Real-time Systems

A real-time system must respond to events within specified time constraints. Characteristics:

**Hard real-time:**
- Missing deadline is catastrophic
- Example: Flight control, anti-lock brakes
- Must guarantee timing

**Soft real-time:**
- Missing deadline degrades performance
- Example: Video streaming, user interfaces
- Statistical timing acceptable

**Control systems are typically hard real-time:**
- Control loop must execute periodically
- Missing deadlines causes instability
- Timing jitter affects performance

**Real-time operating system (RTOS):**

Provides:
- Deterministic scheduling
- Priority-based task execution
- Interrupt handling
- Timing services (clocks, timers)
- Inter-task communication
- Synchronization primitives

Examples: VxWorks, RT-Linux, QNX, FreeRTOS

### Implementation Techniques

**Polling:**
```c
while (1) {
    if (time_to_sample()) {
        y = read_sensor();
        u = control_algorithm(y);
        write_actuator(u);
        wait_until_next_period();
    }
}
```

**Interrupt-driven:**
```c
// Initialization
setup_timer(h);  // Period h

// Interrupt service routine
void timer_ISR() {
    y = read_sensor();
    u = control_algorithm(y);
    write_actuator(u);
}
```

**Task-based:**
```c
// Control task
void control_task() {
    while (1) {
        wait_for_period();  // Suspend until next period
        y = read_sensor();
        u = control_algorithm(y);
        write_actuator(u);
    }
}
```

**Comparison:**
- **Polling:** Simple, but wastes CPU, poor for multiple tasks
- **Interrupt:** Efficient, but limited processing in ISR
- **Task-based:** Most flexible, requires RTOS

### Concurrent Programming

Control systems often have multiple concurrent activities:
- Multiple control loops
- User interface
- Communication
- Diagnostics
- Logging

**Processes/Tasks:**

Each activity is a separate process or task:
```c
task Control_Loop_1() { ... }
task Control_Loop_2() { ... }
task User_Interface() { ... }
task Communication() { ... }
```

**Benefits:**
- Modularity
- Easier to design and test
- Natural expression of concurrent activities

**Challenges:**
- Need synchronization
- Shared resources
- Scheduling
- Timing analysis

### Synchronization and Communication

Tasks need to synchronize and share data.

**Mutual exclusion:**

Protect shared data with locks/semaphores:
```c
// Shared data
float setpoint;
semaphore setpoint_lock;

// Task 1
lock(setpoint_lock);
setpoint = new_value;
unlock(setpoint_lock);

// Task 2
lock(setpoint_lock);
local_setpoint = setpoint;
unlock(setpoint_lock);
```

**Priority inversion:**

Low-priority task holds lock needed by high-priority task. Solution: Priority inheritance or priority ceiling.

**Message passing:**

Tasks communicate via messages:
```c
// Producer
send(queue, message);

// Consumer
message = receive(queue);
```

**Shared memory:**

Direct access with protection:
```c
// Protected shared memory
atomic {
    shared_data = new_value;
}
```

### Periodic Controller Tasks

Control loops are periodic tasks. Key issues:

**Period:**
- Fixed sampling period h
- Specified at design time
- Critical for control performance

**Execution time:**
- Variable due to conditionals, loops
- Worst-case execution time (WCET) must be bounded
- WCET < h for schedulability

**Release time:**
- When task becomes ready to run
- For periodic task: T_k = kh

**Deadline:**
- When task must complete
- Often D_k = (k+1)h

**Response time:**
- Time from release to completion
- Must be < deadline

**Jitter:**

Variation in timing:
- **Release jitter:** Variation in release time
- **Sampling jitter:** Variation in sampling instant
- **Control jitter:** Variation in control output time

Effects of jitter:
- Reduced performance
- Possible instability
- Can be modeled as noise

Reducing jitter:
- High-priority for control tasks
- Minimize interrupt handling
- Avoid blocking
- Use timer-based sampling

### Scheduling

The scheduler decides which task runs when.

**Fixed-priority scheduling:**

Each task has fixed priority. Highest priority ready task runs.

**Rate-Monotonic (RM) scheduling:**
- Priority inversely proportional to period
- Shorter period → higher priority
- Optimal for fixed-priority
- Schedulability test: ∑(C_i/T_i) ≤ n(2^(1/n) - 1)
  where C_i is execution time, T_i is period

**Earliest-Deadline-First (EDF):**
- Dynamic priority based on deadline
- Task with earliest deadline runs
- Optimal for dynamic priority
- Schedulability test: ∑(C_i/T_i) ≤ 1

**Example:**

Three tasks:
- Task 1: T₁ = 10ms, C₁ = 2ms
- Task 2: T₂ = 20ms, C₂ = 5ms
- Task 3: T₃ = 50ms, C₃ = 10ms

Utilization: U = 2/10 + 5/20 + 10/50 = 0.65

RM bound for 3 tasks: 3(2^(1/3) - 1) ≈ 0.78 > 0.65 ✓ Schedulable

**Time-triggered systems:**

All activities scheduled statically in a table:
- Deterministic
- Easy to analyze
- Less flexible

**Event-triggered systems:**

Activities triggered by events:
- More flexible
- Harder to analyze
- Used in RM, EDF

### Summary

Real-time implementation requires:

- **RTOS:** For deterministic timing
- **Concurrent tasks:** For modularity
- **Synchronization:** For shared resources
- **Scheduling:** To meet deadlines
- **Jitter control:** For performance

Key principles:
- Design for worst-case execution time
- Use appropriate scheduling algorithm
- Minimize jitter
- Test timing extensively

Modern control systems are complex real-time systems requiring careful design and implementation.

---

## 13. Controller Timing

The timing of control tasks significantly affects performance. This chapter analyzes timing variations and their effects.

**Timing patterns:**

The execution of a periodic control task can be characterized by:
- **Sampling instant** t_s: When measurement is taken
- **Calculation time** δ: Time to compute control
- **Output instant** t_o = t_s + δ: When control is applied
- **Period** h: Time between samples

**Types of timing:**

1. **Idealized timing:**
   - Zero calculation time (δ = 0)
   - Control applied immediately
   - Rarely achievable in practice

2. **Constant delay:**
   - Fixed calculation time δ
   - Predictable, can compensate
   - t_o = t_s + δ

3. **Variable delay (jitter):**
   - Calculation time varies
   - δ_min ≤ δ ≤ δ_max
   - Harder to compensate

**Effects on control:**

**Phase lag:**
The delay δ introduces phase lag:
$$φ = -ωδ$$

At frequency ω. This reduces phase margin.

**Jitter as noise:**
Timing jitter can be modeled as additive noise in the control loop. The variance is approximately:
$$σ^2 ≈ ω^2σ_δ^2$$

where σ_δ is standard deviation of timing jitter.

**Stability:**
Large or variable delays can destabilize the system. The maximum tolerable delay depends on the loop bandwidth.

**Reducing timing effects:**

1. **Fast execution:**
   - Minimize δ
   - Use efficient algorithms
   - Fast processor

2. **Fixed delay:**
   - Make δ constant
   - Easier to compensate
   - Use deterministic scheduling

3. **Delay compensation:**
   - Include delay in model
   - Design controller with delay
   - Predictor-based control

4. **Time-stamping:**
   - Record exact sampling times
   - Compensate for variations
   - Requires accurate clock

**Implementation patterns:**

**Pattern 1: Sample-Calculate-Output (SCO)**
```
Start of period:
    Sample input → y_k
    Calculate control → u_k
    Output control
    Wait for next period
```
Delay: One calculation time
Jitter: Depends on calculation time variation

**Pattern 2: Output-Sample-Calculate (OSC)**
```
Start of period:
    Output previous control → u_{k-1}
    Sample input → y_k
    Calculate control → u_k
    Wait for next period
```
Delay: One full period plus calculation time
Jitter: Less than SCO if calculation done off-line

**Pattern 3: Parallel execution**
```
High-priority task:
    Sample input → y_k
    Trigger calculation

Low-priority task:
    Calculate control → u_k

High-priority task:
    Output control
```
Delay: Minimized
Jitter: Depends on scheduling

**Example:**

Consider a control loop with:
- Period h = 10ms
- Calculation time δ = 2ms (average), varies ±0.5ms

Pattern SCO:
- Average delay: 2ms
- Jitter: ±0.5ms
- Phase lag at ω = 10 rad/s: -0.02×10 = -0.2 rad ≈ -11.5°

Pattern OSC:
- Average delay: 12ms
- Jitter: minimized by pre-computation
- Phase lag: -0.12×10 = -1.2 rad ≈ -69°

Trade-off: SCO has less delay but more jitter. OSC has more delay but less jitter.

**Synchronization in distributed systems:**

When control system spans multiple computers:
- Clocks must be synchronized
- Network delay must be bounded
- Time-triggered communication helps

**Clock synchronization:**
- NTP (Network Time Protocol): ~1ms accuracy
- PTP (Precision Time Protocol): ~1μs accuracy
- Hardware time-stamping: Best accuracy

### Summary

Controller timing is critical:
- Delays reduce phase margin
- Jitter affects performance
- Implementation pattern matters
- Compensation possible if delay is known and constant

Guidelines:
- Minimize and bound execution time
- Use fixed-priority scheduling
- Compensate for known delays
- Monitor and control jitter
- Synchronize clocks in distributed systems

Careful timing design is essential for high-performance control.

---

## 14. Research Issues

Computer control is a mature field, but research continues in several areas:

**Networked control systems:**
- Control over communication networks
- Handling packet loss and delays
- Co-design of control and communication
- Wireless sensor networks

**Event-based control:**
- Trigger control updates based on events, not time
- Reduce computation and communication
- Maintain performance with less resource usage

**Multi-rate and multi-tasking:**
- Controllers with different sampling rates
- Task scheduling for control systems
- Real-time scheduling theory

**Robust sampled-data control:**
- Robustness to timing variations
- Intersample behavior
- Time-delay systems

**Implementation platforms:**
- Embedded systems and microcontrollers
- Field-programmable gate arrays (FPGAs)
- Multi-core processors
- Hardware acceleration

**Energy-aware control:**
- Minimize energy consumption
- Battery-powered systems
- Energy harvesting

**Security:**
- Cyber-physical security
- Resilient control
- Attack detection and mitigation

**Adaptive and learning:**
- Online parameter estimation
- Adaptive sampling
- Machine learning for control

These research areas address emerging challenges in modern control systems and continue to advance the field of computer control.

---

## Acknowledgments

The authors would like to thank the many colleagues and students who have contributed to the development of computer control theory and practice. This work has been supported by various research grants and industrial collaborations over the years.

---

## Notes and References

This paper provides an overview of computer control. For more detailed treatment, see:

1. Åström, K.J. and Wittenmark, B., Computer-Controlled Systems: Theory and Design, 3rd Edition, Prentice Hall, 1997.

2. Franklin, G.F., Powell, J.D., and Workman, M., Digital Control of Dynamic Systems, 3rd Edition, Addison-Wesley, 1998.

3. Ogata, K., Discrete-Time Control Systems, 2nd Edition, Prentice Hall, 1995.

4. Kuo, B.C., Digital Control Systems, 2nd Edition, Saunders College Publishing, 1992.

5. Wittenmark, B., Åström, K.J., and Årzén, K.-E., "Computer Control: An Overview," IFAC Professional Brief, 2002.

For real-time systems:

6. Liu, J.W.S., Real-Time Systems, Prentice Hall, 2000.

7. Buttazzo, G.C., Hard Real-Time Computing Systems, Springer, 2005.

8. Årzén, K.-E., "Real-Time Control Systems," Lecture Notes, Department of Automatic Control, Lund University, 2020.

For sampled-data control theory:

9. Chen, T. and Francis, B.A., Optimal Sampled-Data Control Systems, Springer, 1995.

10. Dullerud, G.E. and Paganini, F., A Course in Robust Control Theory, Springer, 2000.

---

## Bibliography

[References listed in the paper, covering topics in computer control, real-time systems, sampled-data control, optimization, and practical implementation]

---

## About the Authors

**Björn Wittenmark** is Professor Emeritus at the Department of Automatic Control, Lund University, Sweden. His research interests include computer-controlled systems, adaptive control, and stochastic control.

**Karl Johan Åström** is Professor Emeritus at the Department of Automatic Control, Lund University, Sweden. He is a pioneer in the fields of adaptive control, computer-controlled systems, and control education.

**Karl-Erik Årzén** is Professor at the Department of Automatic Control, Lund University, Sweden. His research interests include real-time control systems, embedded control, and control system implementation.

---

*This document was generated from the IFAC Professional Brief on Computer Control by Wittenmark, Åström, and Årzén.*
