"""
数据加载模块
用于读取和预处理气象数据、作物面积数据和实测灌溉水量数据
"""

import pandas as pd
import numpy as np
import os


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir):
        """
        初始化数据加载器

        Parameters:
        -----------
        data_dir : str
            数据目录路径
        """
        self.data_dir = data_dir

    def _calculate_solar_radiation_from_sunshine(self, dates, sunshine_hours, latitude):
        """
        从日照时数计算太阳辐射
        使用Ångström-Prescott公式：Rs = Ra * (a + b * n/N)
        根据FAO56推荐，a=0.25, b=0.50

        Parameters:
        -----------
        dates : Series
            日期序列
        sunshine_hours : Series
            日照时数 (h)
        latitude : float
            纬度 (度)

        Returns:
        --------
        Series
            太阳辐射 (MJ/m²/day)
        """
        lat_rad = np.radians(latitude)

        # 计算地外辐射和最大日照时数
        Rs_values = []

        for date, n in zip(dates, sunshine_hours):
            # 一年中的第几天
            day_of_year = date.dayofyear

            # 相对日地距离
            b = 2 * np.pi * (day_of_year - 1) / 365
            dr = 1.033 + 0.00867 * np.cos(b)

            # 太阳赤纬
            delta = 0.4093 * np.sin(b)

            # 日出时角
            ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))

            # 地外辐射 Ra (MJ/m²/day)
            Gsc = 0.0820  # 太阳常数
            Ra = (24 * 60 / np.pi) * Gsc * dr * (ws * np.sin(lat_rad) * np.sin(delta) +
                                                  np.cos(lat_rad) * np.cos(delta) * np.sin(ws))

            # 最大可能日照时数 N
            N = 24 * ws / np.pi

            # Ångström-Prescott公式
            a = 0.25
            b_coef = 0.50

            # 防止日照时数超过最大可能值
            n_limited = min(n, N)

            # 计算太阳辐射
            Rs = Ra * (a + b_coef * (n_limited / N))
            Rs_values.append(Rs)

        return pd.Series(Rs_values, index=dates.index)

    def load_weather_data(self, file_path, latitude=38.0):
        """
        加载气象数据

        Parameters:
        -----------
        file_path : str
            气象数据文件路径
        latitude : float
            灌区纬度（用于计算太阳辐射），默认38°N（西北地区）

        Returns:
        --------
        DataFrame
            标准化的气象数据
        """
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 标准化列名（处理可能的不同命名）
        column_mapping = {}
        has_sunshine = False

        for col in df.columns:
            col_lower = str(col).lower().strip()
            if '日期' in col or 'date' in col_lower:
                column_mapping[col] = '日期'
            elif '降雨' in col or 'rain' in col_lower or '降水' in col:
                column_mapping[col] = '降雨量'
            elif '气温' in col or 'temp' in col_lower or '温度' in col:
                column_mapping[col] = '气温'
            elif '湿度' in col or 'humidity' in col_lower or '相对湿度' in col:
                column_mapping[col] = '相对湿度'
            elif '日照' in col or 'sunshine' in col_lower:
                column_mapping[col] = '日照时数'
                has_sunshine = True
            elif '辐射' in col or 'radiation' in col_lower:
                column_mapping[col] = '太阳辐射'
            elif '风速' in col or 'wind' in col_lower:
                column_mapping[col] = '风速'
            elif '气压' in col or 'pressure' in col_lower:
                column_mapping[col] = '气压'

        df = df.rename(columns=column_mapping)

        # 确保日期格式正确
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])

        # 填充缺失值
        # 气象数据缺失时使用前后值的平均
        numeric_cols = ['降雨量', '气温', '相对湿度', '风速']
        if '日照时数' in df.columns:
            numeric_cols.append('日照时数')
        if '太阳辐射' in df.columns:
            numeric_cols.append('太阳辐射')

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')

        # 从日照时数计算太阳辐射（根据FAO56 Ångström-Prescott公式）
        if '日照时数' in df.columns and '太阳辐射' not in df.columns:
            df['太阳辐射'] = self._calculate_solar_radiation_from_sunshine(
                df['日期'], df['日照时数'], latitude
            )

        # 如果太阳辐射仍然缺失，使用默认值（根据季节估算）
        if '太阳辐射' not in df.columns or df['太阳辐射'].isna().all():
            # 使用简化的估算：夏季高，冬季低
            if '日期' in df.columns:
                df['月份'] = df['日期'].dt.month
                # 根据月份估算太阳辐射 (MJ/m²/day)
                radiation_map = {
                    1: 8, 2: 10, 3: 14, 4: 18, 5: 22, 6: 24,
                    7: 23, 8: 21, 9: 17, 10: 13, 11: 9, 12: 7
                }
                df['太阳辐射'] = df['月份'].map(radiation_map)
            else:
                df['太阳辐射'] = 15  # 默认值 MJ/m²/day

        # 如果风速缺失，使用默认值
        if '风速' not in df.columns or df['风速'].isna().all():
            df['风速'] = 2.0  # 默认风速 2 m/s

        # 如果相对湿度缺失，使用默认值
        if '相对湿度' not in df.columns or df['相对湿度'].isna().all():
            df['相对湿度'] = 60.0  # 默认相对湿度 60%

        # 如果气压缺失，使用默认值
        if '气压' not in df.columns or df['气压'].isna().all():
            df['气压'] = 101.3  # 默认大气压 kPa
        else:
            # 将hPa转换为kPa
            df['气压'] = df['气压'] / 10.0

        # 确保所有必需列存在
        required_cols = ['日期', '降雨量', '气温', '相对湿度', '太阳辐射', '风速', '气压']
        for col in required_cols:
            if col not in df.columns:
                print(f"警告: 缺少列 {col}")
        
        return df[required_cols]
    
    def load_crop_area(self, file_path):
        """
        加载作物种植面积数据
        灌溉期（4月-8月）主要作物：春小麦、玉米、经济作物

        Parameters:
        -----------
        file_path : str
            作物面积文件路径

        Returns:
        --------
        dict
            作物面积字典 {作物类型: 面积(亩)}
        """
        # 读取Excel文件（不使用header，因为数据结构特殊）
        df = pd.read_excel(file_path, header=None)

        # 转换为字典
        crop_areas = {}

        # 根据数据结构，第3行（索引2）包含面积数据
        # 结构：夏禾作物(春小麦) | 秋禾作物(玉米) | 经济作物 | 林草地
        if len(df) >= 3:
            # 第3行包含具体作物和面积
            row_data = df.iloc[2]

            # 灌溉期主要作物及其面积
            # 根据数据结构：
            # 列0: 夏禾作物合计 (4846)
            # 列1: 春小麦 (4846)
            # 列2: 秋禾作物合计 (9905)
            # 列3: 玉米 (9905)
            # 列4: 经济作物合计 (36909)
            # 列5+: 各种经济作物

            # 提取灌溉期主要作物
            try:
                # 春小麦（夏禾作物）
                if len(row_data) > 1:
                    spring_wheat = float(row_data.iloc[1])
                    if spring_wheat > 0:
                        crop_areas['春小麦'] = spring_wheat

                # 玉米（秋禾作物）
                if len(row_data) > 3:
                    corn = float(row_data.iloc[3])
                    if corn > 0:
                        crop_areas['玉米'] = corn

                # 经济作物（需要汇总）
                # 经济作物包括：油料、啤酒花、瓜菜果园、孜然、茴香、苜蓿、食葵、洋葱、枸杞、其它
                economic_crops = {}
                economic_crop_names = ['油料', '啤酒花', '瓜菜果园', '孜然', '茴香', '苜蓿', '食葵', '洋葱', '枸杞', '其它']

                # 从列5开始是经济作物的具体品种
                for i, crop_name in enumerate(economic_crop_names):
                    col_idx = 5 + i
                    if col_idx < len(row_data):
                        try:
                            area = float(row_data.iloc[col_idx])
                            if area > 0:
                                economic_crops[crop_name] = area
                        except:
                            pass

                # 将经济作物合并为一个类别（或分别添加）
                # 这里选择分别添加，以便模型可以为不同经济作物设置不同的Kc
                for crop_name, area in economic_crops.items():
                    crop_areas[crop_name] = area

            except Exception as e:
                print(f"警告: 解析作物面积数据时出错: {e}")

        # 如果解析失败，尝试备用方法
        if not crop_areas:
            # 尝试标准的行列结构
            crop_col = None
            area_col = None

            for col in df.columns:
                col_lower = str(col).lower().strip()
                if '作物' in col or 'crop' in col_lower:
                    crop_col = col
                elif '面积' in col or 'area' in col_lower:
                    area_col = col

            if crop_col and area_col:
                for idx, row in df.iterrows():
                    crop_type = str(row[crop_col]).strip()
                    try:
                        area = float(row[area_col])
                        if crop_type and area > 0:
                            crop_areas[crop_type] = area
                    except:
                        continue

        return crop_areas
    
    def load_irrigation_data(self, file_path, crop_areas=None):
        """
        加载实测灌溉水量数据

        Parameters:
        -----------
        file_path : str
            灌溉水量文件路径
        crop_areas : dict, optional
            作物面积字典，用于单位转换（m³转换为mm）

        Returns:
        --------
        DataFrame
            实测灌溉水量数据（单位：mm）
        """
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 标准化列名
        column_mapping = {}

        for col in df.columns:
            col_lower = str(col).lower().strip()
            if '日期' in col or 'date' in col_lower:
                column_mapping[col] = '日期'
            elif '灌溉' in col or 'irrigation' in col_lower or '用水' in col:
                column_mapping[col] = '实测灌溉量'

        df = df.rename(columns=column_mapping)

        # 确保日期格式正确
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])

        # 填充缺失值（灌溉量缺失时设为0）
        if '实测灌溉量' in df.columns:
            df['实测灌溉量'] = df['实测灌溉量'].fillna(0)

            # 单位转换：m³ -> mm
            # 如果提供了作物面积，将m³转换为mm
            if crop_areas is not None and len(crop_areas) > 0:
                # 计算总面积（亩）
                total_area_mu = sum(crop_areas.values())
                # 1亩 = 666.67 m²
                total_area_m2 = total_area_mu * 666.67
                # mm = (m³ / m²) * 1000
                df['实测灌溉量'] = (df['实测灌溉量'] / total_area_m2) * 1000

        return df
    
    def load_calibration_data(self, latitude=38.0):
        """
        加载建模（率定）数据

        Parameters:
        -----------
        latitude : float
            灌区纬度，默认38°N（西北地区）

        Returns:
        --------
        tuple
            (weather_data, crop_areas, irrigation_data)
        """
        calib_dir = os.path.join(self.data_dir, '1 模型测评构模数据')

        # 查找文件
        weather_file = None
        crop_file = None
        irrigation_file = None

        for file in os.listdir(calib_dir):
            if '气象' in file and file.endswith(('.xlsx', '.xls')):
                weather_file = os.path.join(calib_dir, file)
            elif '面积' in file and file.endswith(('.xlsx', '.xls')):
                crop_file = os.path.join(calib_dir, file)
            elif '灌溉' in file and file.endswith(('.xlsx', '.xls')):
                irrigation_file = os.path.join(calib_dir, file)

        weather_data = self.load_weather_data(weather_file, latitude=latitude) if weather_file else None
        crop_areas = self.load_crop_area(crop_file) if crop_file else None
        # 传递crop_areas以进行单位转换
        irrigation_data = self.load_irrigation_data(irrigation_file, crop_areas) if irrigation_file else None

        return weather_data, crop_areas, irrigation_data

    def load_test_data(self, year, latitude=38.0):
        """
        加载测试数据

        Parameters:
        -----------
        year : int
            年份 (2021, 2022, 2024, 2025)
        latitude : float
            灌区纬度，默认38°N（西北地区）

        Returns:
        --------
        tuple
            (weather_data, crop_areas, irrigation_data)
        """
        test_dir = os.path.join(self.data_dir, '2 模型测评率定数据')

        # 查找文件
        weather_file = None
        crop_file = None
        irrigation_file = None

        for file in os.listdir(test_dir):
            if str(year) in file:
                if '气象' in file and file.endswith(('.xlsx', '.xls')):
                    weather_file = os.path.join(test_dir, file)
                elif '面积' in file and file.endswith(('.xlsx', '.xls')):
                    crop_file = os.path.join(test_dir, file)
                elif '灌溉' in file and file.endswith(('.xlsx', '.xls')):
                    irrigation_file = os.path.join(test_dir, file)

        weather_data = self.load_weather_data(weather_file, latitude=latitude) if weather_file else None
        crop_areas = self.load_crop_area(crop_file) if crop_file else None
        # 传递crop_areas以进行单位转换
        irrigation_data = self.load_irrigation_data(irrigation_file, crop_areas) if irrigation_file else None

        return weather_data, crop_areas, irrigation_data

