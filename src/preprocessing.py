"""
Funciones de preprocesamiento reutilizables para el dataset BPC.
Importar desde los notebooks con: from src.preprocessing import *
"""

import pandas as pd
import numpy as np

# Rangos físicos por variable
PHYSICAL_RANGES = {
    'Pres_Succ'        : (0, 400),
    'Pres_Desc'        : (0, 2000),
    'Fluj_Desc'        : (0, 6000),
    'Temp_BBA Acople'  : (20, 200),
    'Temp_BBA Oil-in'  : (20, 200),
    'Temp_BBA Oil-out' : (20, 200),
    'Temp_BBA Carcasa' : (20, 200),
    'Temp_MTR Acople'  : (20, 200),
    'Temp_MTR Libre'   : (20, 200),
    'Temp_Devanado U'  : (20, 210),
    'Temp_Devanado V'  : (20, 210),
    'Temp_Devanado W'  : (20, 210),
    'Corriente L1'     : (0, 450),
    'Corriente L2'     : (0, 450),
    'Corriente L3'     : (0, 450),
    'Voltaje L1-L2'    : (3.8, 4.6),
    'Voltaje L2-L3'    : (3.8, 4.6),
    'Voltaje L1-L3'    : (3.8, 4.6),
    'Potencia'         : (0, 3000),
    'RPM'              : (0, 4000),
    'Pos_axial1'       : (-10, 10),
    'Pos_axial2'       : (-10, 10),
}


def apply_physical_ranges(df: pd.DataFrame, ranges: dict = PHYSICAL_RANGES) -> tuple[pd.DataFrame, int]:
    """Reemplaza valores fuera de rango físico por NaN. Retorna (df_modificado, n_reemplazados)."""
    df = df.copy()
    total_replaced = 0
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            total_replaced += n
    return df, total_replaced


def filter_stops(df: pd.DataFrame, rpm_min: float = 100, pot_min: float = 0) -> tuple[pd.DataFrame, int]:
    """Elimina períodos de paro (RPM < rpm_min o Potencia <= pot_min)."""
    mask_paro = (df['RPM'] < rpm_min) | (df['Potencia'] <= pot_min)
    n_removed = mask_paro.sum()
    return df[~mask_paro].reset_index(drop=True), n_removed


def flag_physical_anomalies(df: pd.DataFrame, pres_umbral: float = 50, flujo_umbral: float = 10) -> pd.DataFrame:
    """Detecta y flaggea anomalías físicas imposibles. No elimina filas."""
    df = df.copy()
    # Caso B: potencia+flujo pero sin presión
    mask_b = (df['Potencia'] > 0) & (df['Fluj_Desc'] > flujo_umbral) & (df['Pres_Desc'] < pres_umbral)
    df.loc[mask_b, 'Pres_Desc'] = np.nan
    df['flag_caso_b'] = mask_b.astype(int)

    # Caso C: RPM alto pero sin flujo
    mask_c = (df['RPM'] > 500) & (df['Fluj_Desc'] <= flujo_umbral)
    df['flag_flujo_cero_operando'] = mask_c.astype(int)

    return df


def impute_nans(df: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """Imputa NaN residuales según tipo de variable."""
    df = df.copy()
    temp_cols      = [c for c in df.columns if 'Temp_' in c]
    hidro_cols     = [c for c in ['Pres_Succ', 'Pres_Desc', 'Fluj_Desc'] if c in df.columns]
    corriente_cols = [c for c in ['Corriente L1', 'Corriente L2', 'Corriente L3'] if c in df.columns]
    voltaje_cols   = [c for c in ['Voltaje L1-L2', 'Voltaje L2-L3', 'Voltaje L1-L3'] if c in df.columns]

    for col in temp_cols + hidro_cols:
        if df[col].isnull().any():
            df[col] = df[col].interpolate(method='linear', limit=max_gap, limit_direction='both')

    for col in corriente_cols + voltaje_cols:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables sintéticas relevantes para el clustering."""
    df = df.copy()
    df['Corriente_prom']     = df[['Corriente L1', 'Corriente L2', 'Corriente L3']].mean(axis=1)
    df['Temp_devanado_prom'] = df[['Temp_Devanado U', 'Temp_Devanado V', 'Temp_Devanado W']].mean(axis=1)
    df['TDH_PSI']            = df['Pres_Desc'] - df['Pres_Succ']
    df['Potencia_especifica'] = df['Potencia'] / (df['Fluj_Desc'] + 1e-6)
    return df


def full_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline completo de preprocesamiento."""
    df, n_range   = apply_physical_ranges(df_raw)
    df, n_paros   = filter_stops(df)
    df            = flag_physical_anomalies(df)
    df            = impute_nans(df)
    df            = engineer_features(df)
    print(f'Pipeline completado: {len(df_raw):,} → {len(df):,} filas')
    print(f'  Valores en rango físico corregidos: {n_range:,}')
    print(f'  Registros de paro eliminados      : {n_paros:,}')
    return df
