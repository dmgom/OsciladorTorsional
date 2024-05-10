# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:27:06 2024

@author: mauri
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


VPP= np.array([480, 800, 1200, 1640, 2060, 2420, 2820, 3180, 3680])
radianes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Ajuste lineal
A = np.vstack([radianes, np.ones(len(radianes))]).T
m, c = np.linalg.lstsq(A, VPP, rcond=None)[0]

# Gráfico
residuales = VPP - (m*radianes + c)

std_residuales = np.std(residuales)

# Residuales estandarizados
residuales_estandarizados = residuales / std_residuales

# Creación de la figura y subplots
fig, axs = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [2, 1]})
sigma_VPP=1
sigma_rad=0.05
# Gráfico de VPP vs radianes
axs[0].errorbar(radianes, VPP, yerr=sigma_VPP, xerr=sigma_rad, fmt='o', label='incertidumbre')
axs[0].scatter(radianes, VPP, label='Datos')
axs[0].plot(radianes, m*radianes + c, 'r', label='Ajuste Lineal')
axs[0].set_xlabel('Radianes')
axs[0].set_ylabel('VPP mV')
axs[0].set_title('Ajuste Lineal de VPP en función del ángulo')
axs[0].legend()

# Ecuación del ajuste lineal
equation = f'VPP = {m:.2f} * rads + {c:.2f}'
axs[0].text(0.35, 3000, equation, fontsize=10)

# Gráfico de residuales
axs[1].scatter(radianes, residuales_estandarizados, color='green')
axs[1].axhline(y=0, color='red', linestyle='--')
axs[1].set_xlabel('Radianes')
axs[1].set_ylabel('Residuales estandarizados')
axs[1].set_title('Residuales del ajuste lineal')

plt.tight_layout()
plt.show()

#%%
##Torque mecanico
g = 9.78
masa = np.array([-100, -300, -400,100, 300, 400])/1000
punto_equilibrio = np.array([-0.21, -0.58, -0.72,0.21, 0.56, 0.7])
sigma_torque=10**-4
radio= 0.0127
torque =  -masa * g * radio
sigma_rad=0.05
#starting=[-0.21]
#new_list=[punto_equilibrio[x]-punto_equilibrio[x-1] for x in range(1,len(punto_equilibrio))]
#new_list=starting+new_list
#print(new_list)

error=(((sigma_torque/punto_equilibrio)**2 + (torque*sigma_rad/punto_equilibrio**2)**2))**1/2
#punto_equilibrio=np.array(new_list)
# Ajuste lineal
alpha = np.sqrt(np.mean((punto_equilibrio - np.mean(punto_equilibrio)) ** 2))  # Incertidumbre común
error_pendiente = std_residuales / (np.sqrt(len(punto_equilibrio)) * alpha)

print('Error estándar de la pendiente del ajuste lineal:', error_pendiente)
A = np.vstack([punto_equilibrio, np.ones(len(punto_equilibrio))]).T
m, c = np.linalg.lstsq(A, torque, rcond=None)[0]

# Residuales
residuales = torque - (m * punto_equilibrio + c)
std_residuales = np.std(residuales)

# Residuales estandarizados
residuales_estandarizados = residuales / std_residuales

# Creación de la figura y subplots
fig, axs = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [2, 1]})

# Gráfico de Torque vs Punto de Equilibrio
axs[0].errorbar(punto_equilibrio, torque, yerr=error, fmt='o', label='incertidumbre')
axs[0].scatter(punto_equilibrio, torque, label='Datos')
axs[0].plot(punto_equilibrio, m * punto_equilibrio + c, 'r', label='Ajuste Lineal')
axs[0].set_xlabel('Punto de Equilibrio (radianes)')
axs[0].set_ylabel('Torque (N*m)')
axs[0].set_title('Torque en función del Punto de Equilibrio')
axs[0].legend()

# Ecuación del ajuste lineal
equation = f'Torque = {m:.3f} * rads + {c:.2f}'
axs[0].text(0.0, 0, equation, fontsize=10)

# Gráfico de residuales
axs[1].scatter(punto_equilibrio, residuales_estandarizados, color='green')
axs[1].axhline(y=0, color='red', linestyle='--')
axs[1].set_xlabel('Punto de Equilibrio (radianes)')
axs[1].set_ylabel('Residuales estandarizados')
axs[1].set_title('Residuales del ajuste lineal')

plt.tight_layout()
plt.show()

print('constante de torsion torque mecanico: ', m)
#%%
from scipy.stats import linregress

# Datos
T = np.array([1.24, 1.28, 1.38, 1.44, 1.5, 1.58, 1.64, 1.68])  # Periodo en segundos
N = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Número de masas

# Ajuste lineal T vs N
slope_T, intercept_T, r_value_T, p_value_T, std_err_T = linregress(N, T)
line_T = slope_T * N + intercept_T

# Ajuste lineal T^2 vs N
slope_T2, intercept_T2, r_value_T2, p_value_T2, std_err_T2 = linregress(N, T**2)
line_T2 = slope_T2 * N + intercept_T2

# Gráfico
fig, axs = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1]})
sigma_T=0.01
# Gráfico T vs N
axs[0, 0].errorbar(N,T, yerr=sigma_T, fmt='o', label='incertidumbre')
axs[0, 0].scatter(N, T, label='Datos')
axs[0, 0].plot(N, line_T, color='red', label='Ajuste Lineal')
axs[0, 0].set_xlabel('Número de masas (N)')
axs[0, 0].set_ylabel('T (s)')
axs[0, 0].set_title('T (s) vs N')
axs[0, 0].text(1, 1.5, f'T = {slope_T:.2f} * N + {intercept_T:.2f}')
axs[0, 0].legend()

# Residuales T vs N
residuales_T = T - line_T

std_residuales = np.std(residuales_T)

# Residuales estandarizados
residuales_estandarizadosT = residuales_T / std_residuales

axs[1, 0].scatter(N, residuales_estandarizadosT, color='green')
axs[1, 0].axhline(y=0, color='red', linestyle='--')
axs[1, 0].set_xlabel('Número de masas (N)')
axs[1, 0].set_ylabel('Residuales estandarizados')
axs[1, 0].set_title('Residuales del ajuste lineal T vs N')

# Gráfico T^2 vs N
axs[0, 1].errorbar(N,T**2, yerr=sigma_T, fmt='o', label='incertidumbre')
axs[0, 1].scatter(N, T**2, label='Datos')
axs[0, 1].plot(N, line_T2, color='red', label='Ajuste Lineal')
axs[0, 1].set_xlabel('Número de masas (N)')
axs[0, 1].set_ylabel('T^2 (s^2)')
axs[0, 1].set_title('T^2 (s^2) vs N')
axs[0, 1].text(1, 2, f'T^2 = {slope_T2:.2f} * N + {intercept_T2:.2f}')
axs[0, 1].legend()

# Residuales T^2 vs N
residuales_T2 = T**2 - line_T2
std_residuales = np.std(residuales_T2)

# Residuales estandarizados
residuales_estandarizadosT2 = residuales_T2 / std_residuales

axs[1, 1].scatter(N, residuales_estandarizadosT2, color='green')
axs[1, 1].axhline(y=0, color='red', linestyle='--')
axs[1, 1].set_xlabel('Número de masas (N)')
axs[1, 1].set_ylabel('Residuales estandarizados')
axs[1, 1].set_title('Residuales del ajuste lineal T^2 vs N')

plt.tight_layout()
plt.show()

##Calculo Inercia y constante

Masa_brasa = 0.2145 # kg
R1 = 0.021  # metros
R2 = 0.047  # metros
kappa = 0.058  # arbitratio
pi=np.pi

sigma_masa=0.0001
sigma_R1=0.001
sigma_R2=0.001
# Cálculo de la inercia
inercia = intercept_T2 * kappa / (4 * pi**2)

# Cálculo de la constante de torsión
constante_torsion = (pi**2 * Masa_brasa * (R1**2 + R2**2)) / (2 * slope_T2)


####
alpha = np.sqrt(np.mean((T ** 2 - np.mean(T ** 2)) ** 2))
# Error de la pendiente utilizando incertidumbre común
error_pendiente = std_err_T2 / alpha
error_pendiente1 = std_residuales /np.sqrt(len(N))
error_interseccion1= error_pendiente1*np.sqrt(np.mean(N**2))
# Error de la intersección utilizando incertidumbre común
error_interseccion = std_err_T2 * np.sqrt(np.mean(N ** 2)) / alpha

print('Incertidumbre de la pendiente del ajuste lineal T^2 vs N:', error_pendiente, error_pendiente1)
print('Incertidumbre de la intersección del ajuste lineal T^2 vs N:', error_interseccion, error_interseccion1)

sigma_inercia= error_interseccion


sigma_kappa= (pi**2 * Masa_brasa * (R1**2 + 2*R2))*sigma_R2/(2*slope_T2)
sigma_kappa+=((pi**2 * Masa_brasa * (2*R1 + R2**2))*sigma_R2/(2 * slope_T2))**2
sigma_kappa+= ((pi**2 * (R1**2 + R2**2))*sigma_masa/(2 * slope_T2))**2
sigma_kappa+= (pi**2 *(R1**2 + R2**2)*error_pendiente /2*slope_T2**2)**2
sigma_kappa= np.sqrt(sigma_kappa)
print("Inercia:", inercia, ' +- ', sigma_inercia, 'intercepto: ', intercept_T2)
print("Constante de Torsión:", constante_torsion, ' +- ', sigma_kappa, 'pendiente: ', error_pendiente)

p, V = np.polyfit(N, T**2, 1, cov=True)

print("x_1: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
print("x_2: {} +/- {}".format(p[1], np.sqrt(V[1][1])))
#%%
##toruqe magnetico
# Datos
posicion_angular = np.array([0.24, 0.42, 0.52, 0.7, 0.8, -0.22, -0.44, -0.5, -0.64, -0.78])  # radianes
corriente = np.array([0.25, 0.5, 0.75, 1, 1.25, -0.25, -0.5, -0.75, -1, -1.25])  # Amperios

# Ajuste lineal
slope, intercept, _, _, _ = linregress(corriente, posicion_angular)
line = slope * corriente + intercept

# Residuales
residuales = posicion_angular - line
std_residuales = np.std(residuales)

# Residuales estandarizados
residuales_estandarizados = residuales / std_residuales
# Creación de la figura y subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})

# Gráfico de radianes vs corriente
axs[0].errorbar(corriente, posicion_angular, yerr=0.05, xerr=0.01, fmt='o', label='incertidumbre')
axs[0].scatter(corriente, posicion_angular, label='Datos')
axs[0].plot(corriente, line, color='red', label='Ajuste Lineal')
axs[0].set_xlabel('Corriente (A)')
axs[0].set_ylabel('Posición Angular (radianes)')
axs[0].set_title('Posición Angular vs Corriente')
axs[0].legend()
equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
axs[0].text(0.5, 0.5, equation_text, transform=axs[0].transAxes, fontsize=12, verticalalignment='bottom')


# Gráfico de residuales con un tamaño menor
axs[1].scatter(corriente, residuales_estandarizados, color='green')
axs[1].axhline(y=0, color='red', linestyle='--')
axs[1].set_xlabel('Corriente (A)')
axs[1].set_ylabel('Residuales estandarizados')
axs[1].set_title('Residuales del ajuste lineal')

plt.tight_layout()
plt.show()

kappa = 0.058  # Arbitrario
c_helmholtz = 3.22/1000  #T/A

# Cálculo de mu
mu = kappa * slope / c_helmholtz
# Incertidumbre común
alpha = np.sqrt(np.mean((posicion_angular - np.mean(posicion_angular)) ** 2))

# Error de la pendiente utilizando incertidumbre común
error_pendiente = std_residuales / alpha

print("Incertidumbre de la pendiente del ajuste lineal:", error_pendiente)

print("Valor de mu:", mu)

#%%
##amortiguamiento factor de calidad
N_mitad=np.array([7,5,4])
Q=4.53*N_mitad
print(Q)
