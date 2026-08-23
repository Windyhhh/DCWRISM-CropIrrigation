"""
日尺度作物需水量与灌溉调控模型 (DCWRISM)
Daily Crop Water Requirement & Irrigation Simulation Model
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DCWRISMModel:
    """日尺度作物需水量与灌溉调控模型"""
    
    def __init__(self, irrigation_efficiency=0.61):
        """
        初始化模型
        
        Parameters:
        -----------
        irrigation_efficiency : float
            综合灌溉水利用系数，默认0.61
        """
        self.irrigation_efficiency = irrigation_efficiency
        
        # 可率定参数 - 作物系数 (Kc)
        # 根据FAO56和西北地区实际情况设置初始值
        self.crop_coefficients = {
            '春小麦': 0.85,      # 小麦生长期Kc
            '玉米': 0.90,        # 玉米生长期Kc
            '油料': 0.75,        # 油菜等油料作物
            '啤酒花': 1.10,      # 啤酒花需水量较大
            '瓜菜果园': 0.80,    # 瓜菜果园
            '孜然': 0.70,        # 孜然
            '茴香': 0.70,        # 茴香
            '苜蓿': 0.85,        # 苜蓿
            '食葵': 0.75,        # 食葵
            '洋葱': 0.75,        # 洋葱
            '枸杞': 0.80,        # 枸杞
            '其它': 0.75,        # 其他经济作物
            '水稻': 1.20,        # 水稻（如果有）
            '棉花': 0.80,        # 棉花（如果有）
            '蔬菜': 0.75,        # 蔬菜（通用）
        }
        
        # 可率定参数 - 有效降雨系数
        self.effective_rainfall_coef = 0.70
        
        # 可率定参数 - 土壤热通量 (MJ/m²/day)
        self.soil_heat_flux = 0.0
        
        # 可率定参数 - 湿度计常数 (kPa/°C)
        self.psychrometric_constant = 0.066
        
    def calculate_saturation_vapor_pressure(self, temp):
        """
        计算饱和水汽压 (kPa)
        
        Parameters:
        -----------
        temp : float or array
            气温 (°C)
        """
        return 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    
    def calculate_slope_vapor_pressure(self, temp):
        """
        计算饱和水汽压曲线斜率 (kPa/°C)
        
        Parameters:
        -----------
        temp : float or array
            气温 (°C)
        """
        return (4098 * self.calculate_saturation_vapor_pressure(temp)) / ((temp + 237.3) ** 2)
    
    def calculate_actual_vapor_pressure(self, temp, relative_humidity):
        """
        计算实际水汽压 (kPa)
        
        Parameters:
        -----------
        temp : float or array
            气温 (°C)
        relative_humidity : float or array
            相对湿度 (%)
        """
        es = self.calculate_saturation_vapor_pressure(temp)
        return es * (relative_humidity / 100.0)
    
    def calculate_extraterrestrial_radiation(self, latitude, day_of_year):
        """
        计算地外辐射 Ra (MJ/m²/day)
        根据FAO56 Chapter 3

        Parameters:
        -----------
        latitude : float
            纬度 (度)
        day_of_year : int
            一年中的第几天 (1-365)
        """
        # 将纬度转换为弧度
        lat_rad = np.radians(latitude)

        # 相对日地距离
        b = 2 * np.pi * (day_of_year - 1) / 365
        dr = 1.033 + 0.00867 * np.cos(b)

        # 太阳赤纬
        delta = 0.4093 * np.sin(b)

        # 日出时角
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))

        # 地外辐射
        Gsc = 0.0820  # 太阳常数 MJ/m²/min
        Ra = (24 * 60 / np.pi) * Gsc * dr * (ws * np.sin(lat_rad) * np.sin(delta) +
                                              np.cos(lat_rad) * np.cos(delta) * np.sin(ws))

        return Ra

    def calculate_solar_radiation_from_sunshine(self, sunshine_hours, Ra, latitude, day_of_year):
        """
        从日照时数计算太阳辐射 Rs (MJ/m²/day)
        使用Ångström-Prescott公式：Rs = Ra * (a + b * n/N)
        根据FAO56推荐，a=0.25, b=0.50

        Parameters:
        -----------
        sunshine_hours : float or array
            实际日照时数 (h)
        Ra : float or array
            地外辐射 (MJ/m²/day)
        latitude : float
            纬度 (度)
        day_of_year : int
            一年中的第几天
        """
        # 计算最大可能日照时数 N
        lat_rad = np.radians(latitude)
        b = 2 * np.pi * (day_of_year - 1) / 365
        delta = 0.4093 * np.sin(b)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        N = 24 * ws / np.pi

        # Ångström-Prescott公式
        # 对于西北地区，使用FAO56推荐的系数
        a = 0.25
        b_coef = 0.50

        # 防止日照时数超过最大可能值
        n = np.minimum(sunshine_hours, N)

        # 计算太阳辐射
        Rs = Ra * (a + b_coef * (n / N))

        return Rs

    def calculate_net_radiation(self, solar_radiation, temp, relative_humidity, atmospheric_pressure):
        """
        计算净辐射 (MJ/m²/day)
        根据FAO56 Chapter 3

        Parameters:
        -----------
        solar_radiation : float or array
            太阳辐射 Rs (MJ/m²/day)
        temp : float or array
            气温 (°C)
        relative_humidity : float or array
            相对湿度 (%)
        atmospheric_pressure : float or array
            大气压 (hPa)
        """
        # 净短波辐射 Rns = (1 - α) * Rs
        # 对于绿色植被，反照率α = 0.23
        alpha = 0.23
        Rns = (1 - alpha) * solar_radiation

        # 净长波辐射 Rnl
        # Rnl = σ * (Tmax^4 + Tmin^4)/2 * (0.34 - 0.14*√ea) * (1.35*Rs/Rso - 0.35)
        # 简化：使用平均气温
        sigma = 4.903e-9  # Stefan-Boltzmann常数 MJ K⁻⁴ m⁻² day⁻¹

        # 计算清晰天空辐射 Rso
        # Rso = (0.75 + 2e-5*z) * Ra，其中z为海拔(m)
        # 简化：Rso ≈ 0.75 * Ra
        # 这里我们用一个简化的方法
        Rso = 0.75 * solar_radiation / 0.75  # 假设Rs/Ra比例

        # 实际水汽压
        es = self.calculate_saturation_vapor_pressure(temp)
        ea = self.calculate_actual_vapor_pressure(temp, relative_humidity)

        # 简化的净长波辐射计算
        # 使用相对湿度和太阳辐射的关系
        Rnl = sigma * ((temp + 273.15) ** 4) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * solar_radiation / (0.75 * solar_radiation) - 0.35)

        # 净辐射
        Rn = Rns - Rnl

        return Rn
    
    def calculate_et0(self, temp, wind_speed, relative_humidity, solar_radiation, atmospheric_pressure=101.3):
        """
        计算参考蒸散量 ET0 (mm/day)
        使用 FAO Penman-Monteith 公式

        Parameters:
        -----------
        temp : float or array
            2米高处日平均气温 (°C)
        wind_speed : float or array
            2米高处风速 (m/s)
        relative_humidity : float or array
            相对湿度 (%)
        solar_radiation : float or array
            太阳辐射 (MJ/m²/day)
        atmospheric_pressure : float or array
            大气压 (kPa)
        """
        # 计算各项参数
        Rn = self.calculate_net_radiation(solar_radiation, temp, relative_humidity, atmospheric_pressure)
        G = self.soil_heat_flux
        es = self.calculate_saturation_vapor_pressure(temp)
        ea = self.calculate_actual_vapor_pressure(temp, relative_humidity)
        delta = self.calculate_slope_vapor_pressure(temp)
        gamma = self.psychrometric_constant

        # FAO Penman-Monteith 公式
        numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (temp + 273)) * wind_speed * (es - ea)
        denominator = delta + gamma * (1 + 0.34 * wind_speed)

        et0 = numerator / denominator

        # 确保 ET0 非负
        et0 = np.maximum(et0, 0)

        return et0
    
    def calculate_etc(self, et0, crop_type):
        """
        计算作物实际蒸散量 ETc (mm/day)
        
        Parameters:
        -----------
        et0 : float or array
            参考蒸散量 (mm/day)
        crop_type : str
            作物类型
        """
        kc = self.crop_coefficients.get(crop_type, 0.80)
        etc = et0 * kc
        return etc
    
    def calculate_effective_rainfall(self, rainfall):
        """
        计算有效降雨量 (mm/day)
        
        Parameters:
        -----------
        rainfall : float or array
            降雨量 (mm)
        """
        pe = rainfall * self.effective_rainfall_coef
        return pe
    
    def calculate_net_irrigation(self, etc, effective_rainfall):
        """
        计算净灌溉用水量 (mm/day)
        
        Parameters:
        -----------
        etc : float or array
            作物实际蒸散量 (mm/day)
        effective_rainfall : float or array
            有效降雨量 (mm)
        """
        net_irrigation = np.maximum(etc - effective_rainfall, 0)
        return net_irrigation
    
    def calculate_gross_irrigation(self, net_irrigation):
        """
        计算毛灌溉用水量 (mm/day)
        
        Parameters:
        -----------
        net_irrigation : float or array
            净灌溉用水量 (mm/day)
        """
        gross_irrigation = net_irrigation / self.irrigation_efficiency
        return gross_irrigation
    
    def simulate_daily(self, weather_data, crop_areas, latitude=38.0):
        """
        模拟日尺度灌溉需水量

        Parameters:
        -----------
        weather_data : DataFrame
            日气象数据，包含列：日期、降雨量、气温、相对湿度、太阳辐射、风速、气压
        crop_areas : dict
            作物种植面积字典 {作物类型: 面积(亩)}
        latitude : float
            灌区纬度（用于太阳辐射计算）

        Returns:
        --------
        DataFrame
            包含每日计算结果的数据框
        """
        results = []

        for idx, row in weather_data.iterrows():
            # 提取气象数据
            date = row['日期']
            temp = row['气温']
            wind_speed = row['风速']
            relative_humidity = row['相对湿度']
            solar_radiation = row['太阳辐射']  # 已经是MJ/m²/day
            rainfall = row['降雨量']
            atmospheric_pressure = row.get('气压', 101.3)  # kPa

            # 计算 ET0
            et0 = self.calculate_et0(temp, wind_speed, relative_humidity, solar_radiation, atmospheric_pressure)

            # 计算有效降雨
            effective_rainfall = self.calculate_effective_rainfall(rainfall)

            # 计算各作物的加权需水量
            total_area = sum(crop_areas.values())
            weighted_etc = 0
            weighted_net_irrigation = 0

            for crop_type, area in crop_areas.items():
                # 计算该作物的 ETc
                etc = self.calculate_etc(et0, crop_type)

                # 计算该作物的净灌溉量
                net_irrigation = self.calculate_net_irrigation(etc, effective_rainfall)

                # 面积加权
                weight = area / total_area
                weighted_etc += etc * weight
                weighted_net_irrigation += net_irrigation * weight

            # 计算毛灌溉量
            gross_irrigation = self.calculate_gross_irrigation(weighted_net_irrigation)

            # 保存结果
            results.append({
                '日期': date,
                'ET0': et0,
                'ETc': weighted_etc,
                '降雨量': rainfall,
                '有效降雨': effective_rainfall,
                '净灌溉量': weighted_net_irrigation,
                '毛灌溉量': gross_irrigation
            })

        return pd.DataFrame(results)
    
    def calculate_10day_irrigation(self, daily_results):
        """
        计算逐旬毛灌溉用水量
        
        Parameters:
        -----------
        daily_results : DataFrame
            日尺度计算结果
        
        Returns:
        --------
        DataFrame
            逐旬毛灌溉用水量
        """
        daily_results = daily_results.copy()
        daily_results['日期'] = pd.to_datetime(daily_results['日期'])
        daily_results['年'] = daily_results['日期'].dt.year
        daily_results['月'] = daily_results['日期'].dt.month
        daily_results['日'] = daily_results['日期'].dt.day
        
        # 定义旬
        def get_10day_period(day):
            if day <= 10:
                return 1  # 上旬
            elif day <= 20:
                return 2  # 中旬
            else:
                return 3  # 下旬
        
        daily_results['旬'] = daily_results['日'].apply(get_10day_period)
        
        # 按年月旬分组求和
        period_results = daily_results.groupby(['年', '月', '旬']).agg({
            '毛灌溉量': 'sum',
            '净灌溉量': 'sum',
            '降雨量': 'sum',
            '有效降雨': 'sum'
        }).reset_index()
        
        # 创建旬标识
        period_results['旬标识'] = (period_results['年'].astype(str) + '-' + 
                                   period_results['月'].astype(str).str.zfill(2) + '-' + 
                                   period_results['旬'].astype(str))
        
        return period_results
    
    def evaluate_accuracy(self, simulated, observed):
        """
        评估模型精度
        
        Parameters:
        -----------
        simulated : array-like
            模拟值
        observed : array-like
            实测值
        
        Returns:
        --------
        dict
            包含各项精度指标的字典
        """
        simulated = np.array(simulated)
        observed = np.array(observed)
        
        # 过滤掉观测值为0的数据点（避免除零错误）
        mask = observed > 0
        simulated_filtered = simulated[mask]
        observed_filtered = observed[mask]
        
        if len(observed_filtered) == 0:
            return {
                '相对误差平均值(%)': np.nan,
                'MSE': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan
            }
        
        # 相对误差平均值
        relative_errors = np.abs((simulated_filtered - observed_filtered) / observed_filtered) * 100
        relative_error_avg = np.mean(relative_errors)
        
        # 均方误差 (MSE)
        mse = np.mean((simulated - observed) ** 2)
        
        # 均方根误差 (RMSE)
        rmse = np.sqrt(mse)
        
        # 平均绝对误差 (MAE)
        mae = np.mean(np.abs(simulated - observed))
        
        return {
            '相对误差平均值(%)': relative_error_avg,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }

