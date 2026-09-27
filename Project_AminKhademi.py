
from math import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

K = 10 ** 6 # zarib tabdil vahed if Q in g/s => ug/m3

def pollution_concentration(stack_params, x, y, z, zref, uref, Hm, Ta, stability_category, type_area, s=None):
    p_peterson_index_dic = {
        ('A', 'R'): 0.07, ('A', 'U'): 0.15, ('B', 'R'): 0.07, ('B', 'U'): 0.15,
        ('C', 'R'): 0.1, ('C', 'U'): 0.2, ('D', 'R'): 0.15, ('D', 'U'): 0.25,
        ('E', 'R'): 0.35, ('E', 'U'): 0.3, ('F', 'R'): 0.35, ('F', 'U'): 0.3
    }
    p = p_peterson_index_dic[(stability_category, type_area)]
    
    sigma_y_rural_dic = {'A':(24.167, 2.5334), 'B':(18.333, 1.8096), 'C':(12.5, 1.0857), 'D':(8.333, 0.72382), 'E':(6.25, 0.54287), 'F':(4.1667, 0.36191)}
    sigma_y_urban_dic = {'A':0.32, 'B':0.32, 'C':0.22, 'D':0.16, 'E':0.11, 'F':0.11}
    sigma_z_urban_dic = {'A':(0.24, 0.001, 0.5), 'B':(0.24, 0.001, 0.5), 'C':(0.2, 0, 0), 'D':(0.14, 0.0003, -0.5), 'E':(0.08, 0.0015, -0.5), 'F':(0.08, 0.0015, -0.5)}

    distancex = x - stack_params[0]
    distancey = y - stack_params[1]
    
    # بررسی مسافت - فقط در جهت باد (distancex) نباید منفی باشد
    if distancex <= 0:
        print(f"Warning: Point ({x},{y}) is upwind from stack at ({stack_params[0]},{stack_params[1]})")
        return 0.0
        
    # محاسبه مسافت کل از منبع
    distance = sqrt(distancex**2 + distancey**2)
    
    if type_area == 'R':
        c, d = sigma_y_rural_dic[stability_category]
        # استفاده از مسافت واقعی به جای distancex
        TH = 0.017453293 * (c - d * log(distance / 1000))
        sigma_y = 465.11628 * distance / 1000 * tan(TH)
        sigma_z = sigma_z_rural(distance / 1000, stability_category)

    elif type_area == 'U':
        a = sigma_y_urban_dic[stability_category]
        b, c, d = sigma_z_urban_dic[stability_category]
        sigma_y = a * distance * pow(1 + 0.0004 * distance, -0.5)
        sigma_z = b * distance * pow(1 + c * distance, d)

    # محاسبه سرعت باد در ارتفاع دودکش
    us = uref * pow(stack_params[2] / zref, p)

    # بررسی حداقل سرعت باد
    if us < 0.1:
        us = 0.1
        print(f"Warning: Wind speed too low, set to minimum 0.1 m/s")

    # محاسبه ارتفاع مؤثر دودکش
    if stack_params[4] < 1.5 * us:
        h_prime_s = stack_params[2] + 2 * stack_params[3] * (stack_params[4] / us - 1.5)
    else:
        h_prime_s = stack_params[2]

    dT = stack_params[5] - Ta
    
    # بررسی منطقی بودن دماها
    if stack_params[5] <= 0 or Ta <= 0:
        print(f"Error: Invalid temperature values")
        return 0.0
    
    # محاسبه پارامترهای شناوری
    Fm = stack_params[4] ** 2 * stack_params[3] ** 2 * Ta / 4 / stack_params[5]
    if dT != 0:
        Fb = 9.81 * stack_params[4] * stack_params[3] ** 2 * dT / 4 / stack_params[5]
    else:
        Fb = 0.0

    # محاسبه ارتفاع بلند شدن پلوم
    if stability_category in 'EF':
        if s is None or s <= 0:
            s = 0.01  # مقدار پیش‌فرض
        
        dTc = 0.01958 * stack_params[5] * stack_params[4] * sqrt(s)

        if dT >= dTc:
            dh = 2.6 * pow(abs(Fb) / us / s, 1/3)
            xf = 2.075 * us / sqrt(s)
        else:
            dh = 1.5 * pow(Fm / us / sqrt(s), 1/3)
            xf = 0.5 * pi * us / sqrt(s)

    elif stability_category in 'ABCD':
        if Fb < 55:
            dTc = 0.0297 * stack_params[5] * pow(stack_params[4], 1/3) / pow(stack_params[3], 2/3)

            if dT >= dTc:
                dh = 21.425 * pow(abs(Fb), 0.75) / us
                xf = 49 * pow(abs(Fb), 0.625)
            else:
                dh = 3 * stack_params[3] * stack_params[4] / us
                if Fb == 0.0:
                    dh = 3 * stack_params[3] * stack_params[4] / us
                    xf = 4 * stack_params[3] * pow(stack_params[4] + 3 * us, 2) / stack_params[4] / us
                else:
                    xf = 49 * pow(abs(Fb), 0.625)
        else:
            dTc = 0.00575 * stack_params[5] * pow(stack_params[4], 2/3) / pow(stack_params[3], 1/3)
            xf = 119 * pow(abs(Fb), 0.4)
            if dT >= dTc:
                dh = 38.71 * pow(abs(Fb), 0.6) / us
            else:
                dh = 3 * stack_params[3] * stack_params[4] / us

    # بررسی برای مسافه کمتر از xf
    if distance < xf:
        if dT >= dTc:
            dh = 1.6 * pow(abs(Fb), 1/3) * pow(distance, 2/3) / us
        else:
            beta_j = 1/3 + us / stack_params[4]
            if stability_category in 'ABCD':
                dh = pow(3 * Fm * distance / beta_j**2 / us**2, 1/3)
            elif stability_category in 'EF':
                if s > 0:
                    sin_arg = distance * sqrt(s) / us
                    dh = 1.6 * pow(3 * Fm * sin(sin_arg) / beta_j / us / sqrt(s), 1/3)

    he = h_prime_s + dh
    sigma_ye = sqrt(sigma_y**2 + (dh / 3.5)**2)
    sigma_ze = sqrt(sigma_z**2 + (dh / 3.5)**2)

    # بررسی صفر نبودن sigma ها
    if sigma_ye <= 0 or sigma_ze <= 0:
        print(f"Warning: Invalid sigma values at distance {distance}")
        return 0.0

    # محاسبه فاکتور واپاشی
    D = exp(stack_params[7] * (-1) * distance / us)

    # محاسبه توزیع عمودی
    if (sigma_ze / Hm) > 1.6:
        V = sqrt(2 * pi) * sigma_ze / Hm
    else:
        term1 = exp(-0.5 * pow((z - he) / sigma_ze, 2))
        term2 = exp(-0.5 * pow((z + he) / sigma_ze, 2))
        V = term1 + term2

    # محاسبه غلظت نهایی
    try:
        exp_term = exp(-0.5 * pow(distancey / sigma_ye, 2))
        Cxyz = stack_params[6] * K * V * D / (2 * pi * us * sigma_ye * sigma_ze) * exp_term
        
        if Cxyz < 0 or isnan(Cxyz) or isinf(Cxyz):
            return 0.0
            
        return Cxyz
    except:
        print(f"Error in calculation at point ({x},{y})")
        return 0.0


def sigma_z_rural(x, stability_category):
    if stability_category == 'A':
        if x < 0.1:
            a, b = 122.8, 0.9447
        elif x < 0.15:
            a, b = 158.08, 1.0542
        elif x < 0.2:
            a, b = 170.22, 1.0932
        elif x < 0.25:
            a, b = 179.52, 1.1262
        elif x < 0.3:
            a, b = 217.41, 1.2644
        elif x < 0.4:
            a, b = 258.89, 1.4094
        elif x < 0.5:
            a, b = 346.75, 1.7283
        elif x < 3.11:
            a, b = 453.85, 2.1166
        else:
            a, b = 2000, 1

    elif stability_category == 'B':
        if x < 0.2:
            a, b = 90.673, 0.93198
        elif x < 0.4:
            a, b = 98.483, 0.98332
        else:
            a, b = 109.3, 1.0971

    elif stability_category == 'C':
        a, b = 61.141, 0.91465

    elif stability_category == 'D':
        if x < 0.3:
            a, b = 34.459, 0.86974
        elif x < 1:
            a, b = 32.093, 0.81066
        elif x < 3:
            a, b = 32.093, 0.64403
        elif x < 10:
            a, b = 33.504, 0.60486
        elif x < 30:
            a, b = 36.65, 0.56589
        else:
            a, b = 44.053, 0.51179

    elif stability_category == 'E':
        if x < 0.1:
            a, b = 24.26, 0.8366
        elif x < 0.3:
            a, b = 23.331, 0.81956
        elif x < 1:
            a, b = 21.628, 0.7566
        elif x < 2:
            a, b = 21.628, 0.63077
        elif x < 4:
            a, b = 22.534, 0.57154
        elif x < 10:
            a, b = 24.703, 0.50527
        elif x < 20:
            a, b = 26.97, 0.46713
        elif x < 40:
            a, b = 35.42, 0.37615
        else:
            a, b = 47.618, 0.29592

    elif stability_category == 'F':
        if x < 0.2:
            a, b = 15.209, 0.81558
        elif x < 0.7:
            a, b = 14.457, 0.78407
        elif x < 1:
            a, b = 13.953, 0.68465
        elif x < 2:
            a, b = 13.953, 0.63227
        elif x < 3:
            a, b = 14.823, 0.54503
        elif x < 7:
            a, b = 16.187, 0.4649
        elif x < 15:
            a, b = 17.836, 0.41507
        elif x < 30:
            a, b = 22.651, 0.32681
        elif x < 60:
            a, b = 27.074, 0.27436
        else:
            a, b = 34.219, 0.21716

    sigma_z = a * pow(x, b)

    if stability_category in 'ABC' and sigma_z > 5000:
        sigma_z = 5000

    return sigma_z


def create_visualizations(x_list, y_list, p_list, length, width, stack_parameters):
    """Different graphics show"""
    
    # تنظیمات کلی
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # نمایش اول: نقاط با مقادیر عددی
    ax1 = plt.subplot(2, 3, 1)
    scatter = plt.scatter(x_list, y_list, c=p_list, s=80, alpha=0.7, cmap='YlOrRd', marker='s')
    
    # اضافه کردن موقعیت دودکش‌ها
    for i, stack in enumerate(stack_parameters):
        plt.plot(stack[0], stack[1], 'ko', markersize=15, markerfacecolor='black', 
                markeredgecolor='white', markeredgewidth=2)
        plt.annotate(f'Stack {i+1}', (stack[0], stack[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, color='white', weight='bold')
    
    plt.colorbar(scatter, label='Concentration (μg/m³)')
    plt.xlabel('Coord X (m)')
    plt.ylabel('Coord Y (m)')
    plt.title('Point show concentration')
    plt.grid(True, alpha=0.3)
    
    # نمایش دوم: نمودار کانتور
    ax2 = plt.subplot(2, 3, 2)
    if len(set(x_list)) > 1 and len(set(y_list)) > 1:
        # تبدیل به شبکه منظم
        x_unique = sorted(set(x_list))
        y_unique = sorted(set(y_list))
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X)
        
        for i, xi in enumerate(x_unique):
            for j, yi in enumerate(y_unique):
                # پیدا کردن مقدار مربوط به این نقطه
                for k, (x_val, y_val, p_val) in enumerate(zip(x_list, y_list, p_list)):
                    if x_val == xi and y_val == yi:
                        Z[j, i] = p_val
                        break
        
        contour = plt.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.8)
        plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour, label='Concentration (μg/m³)')
        
        # اضافه کردن موقعیت دودکش‌ها
        for i, stack in enumerate(stack_parameters):
            plt.plot(stack[0], stack[1], 'ko', markersize=12, markerfacecolor='blue', 
                    markeredgecolor='white', markeredgewidth=2)
            plt.annotate(f'S{i+1}', (stack[0], stack[1]), 
                        xytext=(0, 0), textcoords='offset points',
                        fontsize=8, color='white', weight='bold', ha='center', va='center')
    
    plt.xlabel('Coord X (m)')
    plt.ylabel('coord Y (m)')
    plt.title('Concentration contour chart')
    plt.grid(True, alpha=0.3)
    
    # نمایش سوم: نمودار سطح سه‌بعدی
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    if len(set(x_list)) > 1 and len(set(y_list)) > 1:
        surf = ax3.plot_surface(X, Y, Z, cmap='YlOrRd', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax3.set_xlabel('Coord X (m)')
        ax3.set_ylabel('Coord Y (m)')
        ax3.set_zlabel('Concentration (μg/m³)')
        ax3.set_title('3D display of pollution distribution')
        plt.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, label='Concentration (μg/m³)')
    
    # نمایش چهارم: نمودار heat map
    ax4 = plt.subplot(2, 3, 4)
    if len(set(x_list)) > 1 and len(set(y_list)) > 1:
        heatmap = plt.imshow(Z, extent=[min(x_unique), max(x_unique), 
                                       min(y_unique), max(y_unique)], 
                           origin='lower', cmap='YlOrRd', aspect='auto', alpha=0.8)
        plt.colorbar(heatmap, label='Concentration (μg/m³)')
        
        # اضافه کردن موقعیت دودکش‌ها
        for i, stack in enumerate(stack_parameters):
            plt.plot(stack[0], stack[1], 'wo', markersize=10, markerfacecolor='blue', 
                    markeredgecolor='white', markeredgewidth=2)
    
    plt.xlabel('Coord X (m)')
    plt.ylabel('Coord Y (m)')
    plt.title('SHow Heat Map')
    plt.grid(True, alpha=0.3)
    
    # نمایش پنجم: نمودار میله‌ای بالاترین مقادیر
    ax5 = plt.subplot(2, 3, 5)
    # پیدا کردن 10 نقطه با بالاترین غلظت
    sorted_indices = sorted(range(len(p_list)), key=lambda i: p_list[i], reverse=True)[:10]
    top_concentrations = [p_list[i] for i in sorted_indices]
    top_positions = [f'({x_list[i]},{y_list[i]})' for i in sorted_indices]
    
    bars = plt.bar(range(len(top_concentrations)), top_concentrations, 
                   color='orangered', alpha=0.7, edgecolor='darkred')
    plt.xlabel('Points position')
    plt.ylabel('Concentration (μg/m³)')
    plt.title('Top 10 with Max Concentration')
    plt.xticks(range(len(top_positions)), top_positions, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # اضافه کردن مقادیر روی میله‌ها
    for i, (bar, value) in enumerate(zip(bars, top_concentrations)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # نمایش ششم: آمار توصیفی
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # محاسبه آمار
    non_zero_values = [p for p in p_list if p > 0]
    total_points = len(p_list)
    non_zero_points = len(non_zero_values)
    max_conc = max(p_list) if p_list else 0
    min_conc = min(p_list) if p_list else 0
    avg_conc = sum(p_list) / len(p_list) if p_list else 0
    avg_non_zero = sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0
    
    # نمایش آمار
    stats_text = f"""
    General stattistic   :
    
    total points: {total_points}
    non zero points: {non_zero_points}
    total points: {(non_zero_points/total_points)*100:.1f}%
    
    Max Concentration: {max_conc:.3f} μg/m³
    Min concentration: {min_conc:.3f} μg/m³
    Ttal average: {avg_conc:.3f} μg/m³
    Averagr nonzero: {avg_non_zero:.3f} μg/m³
    
    Total stackes: {len(stack_parameters)}
    Region area {length}×{width} m
    """
    
    plt.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# ورودی‌های کاربر
print("===    Air pollution model ===\n")

length = int(input('  Lenth mesh (m): '))
width = int(input('  width mesh (m): '))
mesh_x = int(input('mesh size X: '))
mesh_y = int(input('mesh size Y: '))
num_stacks = int(input('number of stacks : '))
z = int(input(' Assessment Height (CONS) : '))

stack_parameters = []
for i in range(num_stacks):
    print(f"\n--- Stack {i+1} ---")
    stack_x = int(input('Position X (m): '))
    stack_y = int(input('Position Y (m): '))
    hs = float(input('Stack height (m): '))
    ds = float(input('Stack diameter (m): '))
    vs = float(input('Gas velosity (m/s): '))
    Ts = float(input('Gas temp (K): '))
    Q = float(input('flow rate Q (g/s): '))
    saay = float(input('Psi ψ (s^-1): '))
    stack_parameters.append((stack_x, stack_y, hs, ds, vs, Ts, Q, saay))

print("\n---  Atmospheric Parameters ---")
zref = float(input('Z ref  (m): '))
uref = float(input('u ref   (m/s): '))
Hm = float(input('Mixing layer height (m): '))
Ta = float(input('Ambiant temp (K): '))
stability_category = input(' Stability class (A to F): ').upper()

s = None
if stability_category in 'EF':
    s = float(input('Stability parameter  s (s^-2): '))

type_area = input('Region type (R: rural, U: urban): ').upper()

print("\n===  Calculating... ===")

# محاسبه غلظت در نقاط شبکه
x_list, y_list, p_list = [], [], []

for j in range(0, int(width/mesh_y) + 1):
    y_coord = j * mesh_y
    for i in range(0, int(length/mesh_x) + 1):
        x_coord = i * mesh_x
        
        total_pollution = 0.0
        for stack_params in stack_parameters:
            conc = pollution_concentration(stack_params, x_coord, y_coord, z, zref, uref, Hm, Ta, stability_category, type_area, s)
            total_pollution += conc
        
        print(f'Point ({x_coord}, {y_coord}): {total_pollution:.6f} μg/m³')
        x_list.append(x_coord)
        y_list.append(y_coord)
        p_list.append(round(total_pollution, 6))

print("\n=== Calculation finished ===")
print(f"   Number of point calculated: {len(p_list)}")
print(f"   Points with non zero concentration: {sum(1 for p in p_list if p > 0)}")

# ایجاد نمایش‌های گرافیکی
create_visualizations(x_list, y_list, p_list, length, width, stack_parameters)