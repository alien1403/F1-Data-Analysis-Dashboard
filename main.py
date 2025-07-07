import os
import pandas as pd
import numpy as np
import fastf1
import fastf1.plotting
from flask import Flask, render_template, request, jsonify, send_file
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import logging
from io import BytesIO
from scipy.stats import linregress

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
if not os.path.exists("cache_dir"):
    os.makedirs("cache_dir")
fastf1.Cache.enable_cache('cache_dir')

YEARS = [2025, 2024, 2023]
GRAND_PRIX_LIST = [
    'Australia', 'Bahrain', 'Saudi Arabia', 'Japan', 'China', 'Miami',
    'Monaco', 'Canada', 'Spain', 'Austria', 'Great Britain', 'Hungary',
    'Belgium', 'Netherlands', 'Italy', 'Singapore', 'United States',
    'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

def load_session(year, grand_prix, session_type='Q'):
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(telemetry=True, laps=True, weather=True)
        logger.info(f"Loaded session for {year} {grand_prix} {session_type}")
        return session, None
    except Exception as e:
        logger.warning(f"Failed to load {year} {grand_prix}: {str(e)}")
        if year != 2023:
            try:
                session = fastf1.get_session(2023, grand_prix, session_type)
                session.load(telemetry=True, laps=True, weather=True)
                logger.info(f"Fallback to 2023 {grand_prix} successful")
                return session, f"Warning: Using 2023 data for {grand_prix} as {year} data is unavailable."
            except Exception as e2:
                logger.error(f"Failed to load 2023 {grand_prix}: {str(e2)}")
        return None, f"Error: Could not load data for {grand_prix} in {year}."

def get_empty_plot(title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="No Data",
        yaxis_title="No Data",
        template="plotly_dark",
        annotations=[dict(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)]
    )
    return pio.to_json(fig)

def get_driver_comparison_plot(session, drivers, metric='Speed'):
    try:
        fig = go.Figure()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan']
        for i, driver in enumerate(drivers):
            fast_lap = session.laps.pick_driver(driver).pick_fastest()
            if fast_lap.empty:
                logger.warning(f"No fast lap data for {driver}")
                continue
            car_data = fast_lap.get_car_data()
            if car_data.empty:
                logger.warning(f"No telemetry data for {driver}")
                continue
            fig.add_trace(go.Scatter(
                x=car_data['Time'].dt.total_seconds(),
                y=car_data[metric],
                mode='lines',
                name=f"{driver} {metric}",
                line=dict(color=colors[i % len(colors)])
            ))
            if metric == 'Speed':
                fig.add_trace(go.Scatter(
                    x=car_data['Time'].dt.total_seconds(),
                    y=car_data['Throttle'],
                    mode='lines',
                    name=f"{driver} Throttle",
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    yaxis='y2'
                ))
        
        if not fig.data:
            logger.warning("No data for driver comparison plot")
            return get_empty_plot(f"{metric} Comparison")
        
        yaxis_title = "Speed (km/h)" if metric == 'Speed' else "RPM"
        fig.update_layout(
            title=f"{metric} Comparison",
            xaxis_title="Time (s)",
            yaxis=dict(title=yaxis_title, side="left"),
            yaxis2=dict(title="Throttle (%)", side="right", overlaying="y") if metric == 'Speed' else None,
            template="plotly_dark",
            showlegend=True
        )
        logger.info(f"Generated {metric} comparison plot for {drivers}")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_driver_comparison_plot: {str(e)}")
        return get_empty_plot(f"{metric} Comparison")

def get_sector_time_plot(session, drivers):
    try:
        sector_times = []
        for driver in drivers:
            fast_lap = session.laps.pick_driver(driver).pick_fastest()
            if not fast_lap.empty:
                sector_times.append({
                    'Driver': driver,
                    'Sector1': fast_lap['Sector1Time'].total_seconds() if pd.notna(fast_lap['Sector1Time']) else 0,
                    'Sector2': fast_lap['Sector2Time'].total_seconds() if pd.notna(fast_lap['Sector2Time']) else 0,
                    'Sector3': fast_lap['Sector3Time'].total_seconds() if pd.notna(fast_lap['Sector3Time']) else 0
                })
        
        if not sector_times:
            logger.warning("No sector time data available")
            return get_empty_plot("Sector Times Comparison")
        
        df = pd.DataFrame(sector_times)
        fig = go.Figure()
        for sector in ['Sector1', 'Sector2', 'Sector3']:
            fig.add_trace(go.Bar(
                x=df['Driver'],
                y=df[sector],
                name=sector
            ))
        
        fig.update_layout(
            title="Sector Times Comparison",
            xaxis_title="Driver",
            yaxis_title="Time (s)",
            barmode='group',
            template="plotly_dark"
        )
        logger.info("Generated sector time plot")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_sector_time_plot: {str(e)}")
        return get_empty_plot("Sector Times Comparison")

def get_tire_strategy_plot(session, driver):
    try:
        laps = session.laps.pick_driver(driver)
        laps = laps.dropna(subset=['Compound'])
        laps['LapNumber'] = laps['LapNumber'].astype(int)
        
        if laps.empty:
            logger.warning(f"No tire data for {driver}")
            return get_empty_plot(f"Tire Strategy for {driver}")
        
        compounds = laps['Compound'].unique()
        fig = go.Figure()
        for compound in compounds:
            compound_laps = laps[laps['Compound'] == compound]
            fig.add_trace(go.Bar(
                x=compound_laps['LapNumber'],
                y=[1] * len(compound_laps),
                name=compound,
                text=compound_laps['Compound']
            ))
        
        fig.update_layout(
            title=f"Tire Strategy for {driver}",
            xaxis_title="Lap Number",
            yaxis_title="Lap Count",
            barmode='stack',
            template="plotly_dark"
        )
        logger.info(f"Generated tire strategy plot for {driver}")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_tire_strategy_plot: {str(e)}")
        return get_empty_plot(f"Tire Strategy for {driver}")

def get_lap_time_distribution(session):
    try:
        laps = session.laps
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        laps = laps.dropna(subset=['LapTimeSeconds'])
        
        if laps.empty:
            logger.warning("No lap time data available")
            return get_empty_plot("Lap Time Distribution")
        
        fig = go.Figure()
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver]['LapTimeSeconds']
            fig.add_trace(go.Violin(
                y=driver_laps,
                name=driver,
                box_visible=True,
                meanline_visible=True,
                points='all',
                jitter=0.05,
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Lap Time Distribution",
            xaxis_title="Driver",
            yaxis_title="Lap Time (s)",
            template="plotly_dark",
            showlegend=True
        )
        logger.info("Generated lap time distribution plot")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_lap_time_distribution: {str(e)}")
        return get_empty_plot("Lap Time Distribution")

def get_gear_shift_plot(session, driver):
    try:
        fast_lap = session.laps.pick_driver(driver).pick_fastest()
        telemetry = fast_lap.get_telemetry()
        telemetry = telemetry[['Time', 'nGear']].dropna()
        
        if telemetry.empty:
            logger.warning(f"No gear data for {driver}")
            return get_empty_plot(f"Gear Shift Analysis for {driver}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=telemetry['Time'].dt.total_seconds(),
            y=telemetry['nGear'],
            mode='lines+markers',
            name=f"{driver} Gear",
            line=dict(color='purple')
        ))
        
        fig.update_layout(
            title=f"Gear Shift Analysis for {driver}",
            xaxis_title="Time (s)",
            yaxis_title="Gear",
            template="plotly_dark"
        )
        logger.info(f"Generated gear shift plot for {driver}")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_gear_shift_plot: {str(e)}")
        return get_empty_plot(f"Gear Shift Analysis for {driver}")

def get_telemetry_heatmap(session, driver, metric='Speed'):
    try:
        fast_lap = session.laps.pick_driver(driver).pick_fastest()
        telemetry = fast_lap.get_telemetry()
        telemetry = telemetry[['X', 'Y', metric]].dropna()
        
        if telemetry.empty:
            logger.warning(f"No {metric} data for {driver}")
            return get_empty_plot(f"{metric} Heatmap for {driver}")
        
        fig = px.scatter(
            telemetry,
            x='X',
            y='Y',
            color=metric,
            color_continuous_scale='Viridis',
            title=f"{metric} Heatmap for {driver}",
            labels={'X': 'Track X', 'Y': 'Track Y', metric: f"{metric} ({'km/h' if metric == 'Speed' else '%'})"}
        )
        
        fig.update_layout(template='plotly_dark')
        logger.info(f"Generated {metric} heatmap for {driver}")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_telemetry_heatmap: {str(e)}")
        return get_empty_plot(f"{metric} Heatmap for {driver}")

def get_track_dominance_map(session):
    try:
        drivers = session.results['Abbreviation'].tolist()
        telemetry_data = []
        for driver in drivers:
            fast_lap = session.laps.pick_driver(driver).pick_fastest()
            if not fast_lap.empty:
                telemetry = fast_lap.get_telemetry()[['X', 'Y', 'Time']].dropna()
                telemetry['Driver'] = driver
                telemetry_data.append(telemetry)
        
        if not telemetry_data:
            logger.warning("No telemetry data for track dominance")
            return get_empty_plot("Track Dominance Map")
        
        all_telemetry = pd.concat(telemetry_data)
        all_telemetry['Time'] = all_telemetry['Time'].dt.total_seconds()
        
        dominance = all_telemetry.groupby(['X', 'Y'])['Time'].idxmin()
        dominance_data = all_telemetry.loc[dominance][['X', 'Y', 'Driver']]
        
        fig = px.scatter(
            dominance_data,
            x='X',
            y='Y',
            color='Driver',
            title="Track Dominance Map",
            labels={'X': 'Track X', 'Y': 'Track Y'}
        )
        
        fig.update_layout(template='plotly_dark')
        logger.info("Generated track dominance map")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_track_dominance_map: {str(e)}")
        return get_empty_plot("Track Dominance Map")

def get_lap_time_trend(session, drivers):
    try:
        fig = go.Figure()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan']
        for i, driver in enumerate(drivers):
            laps = session.laps.pick_driver(driver)
            laps = laps.dropna(subset=['LapTime'])
            laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
            if laps.empty:
                logger.warning(f"No lap data for {driver}")
                continue
            fig.add_trace(go.Scatter(
                x=laps['LapNumber'],
                y=laps['LapTimeSeconds'],
                mode='lines+markers',
                name=f"{driver} Lap Times",
                line=dict(color=colors[i % len(colors)])
            ))
            if len(laps) > 1:
                slope, intercept, _, _, _ = linregress(laps['LapNumber'], laps['LapTimeSeconds'])
                trend_line = intercept + slope * laps['LapNumber']
                fig.add_trace(go.Scatter(
                    x=laps['LapNumber'],
                    y=trend_line,
                    mode='lines',
                    name=f"{driver} Trend",
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))
        
        if not fig.data:
            logger.warning("No data for lap time trend")
            return get_empty_plot("Lap Time Trend")
        
        fig.update_layout(
            title="Lap Time Trend",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (s)",
            template="plotly_dark",
            showlegend=True
        )
        logger.info(f"Generated lap time trend plot for {drivers}")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_lap_time_trend: {str(e)}")
        return get_empty_plot("Lap Time Trend")

def get_driver_consistency_plot(session):
    try:
        laps = session.laps
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        laps = laps.dropna(subset=['LapTimeSeconds'])
        
        if laps.empty:
            logger.warning("No lap time data available")
            return get_empty_plot("Driver Consistency")
        
        fig = go.Figure()
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver]['LapTimeSeconds']
            fig.add_trace(go.Box(
                y=driver_laps,
                name=driver,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Driver Consistency",
            xaxis_title="Driver",
            yaxis_title="Lap Time (s)",
            template="plotly_dark",
            showlegend=True
        )
        logger.info("Generated driver consistency plot")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_driver_consistency_plot: {str(e)}")
        return get_empty_plot("Driver Consistency")

def get_sector_dominance_plot(session):
    try:
        laps = session.laps
        sector_times = []
        for driver in laps['Driver'].unique():
            fast_lap = laps.pick_driver(driver).pick_fastest()
            if not fast_lap.empty:
                sector_times.append({
                    'Driver': driver,
                    'Sector1': fast_lap['Sector1Time'].total_seconds() if pd.notna(fast_lap['Sector1Time']) else np.inf,
                    'Sector2': fast_lap['Sector2Time'].total_seconds() if pd.notna(fast_lap['Sector2Time']) else np.inf,
                    'Sector3': fast_lap['Sector3Time'].total_seconds() if pd.notna(fast_lap['Sector3Time']) else np.inf
                })
        
        if not sector_times:
            logger.warning("No sector time data available")
            return get_empty_plot("Sector Dominance")
        
        df = pd.DataFrame(sector_times)
        dominance = {'Sector1': [], 'Sector2': [], 'Sector3': []}
        for sector in ['Sector1', 'Sector2', 'Sector3']:
            min_time = df[sector].min()
            if min_time != np.inf:
                dominance[sector] = df[df[sector] == min_time]['Driver'].tolist()
        
        fig = go.Figure()
        for sector in ['Sector1', 'Sector2', 'Sector3']:
            counts = pd.Series(dominance[sector]).value_counts()
            fig.add_trace(go.Bar(
                x=counts.index,
                y=counts.values,
                name=sector
            ))
        
        fig.update_layout(
            title="Sector Dominance",
            xaxis_title="Driver",
            yaxis_title="Number of Fastest Sectors",
            barmode='stack',
            template="plotly_dark"
        )
        logger.info("Generated sector dominance plot")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_sector_dominance_plot: {str(e)}")
        return get_empty_plot("Sector Dominance")

def get_track_segment_dominance(session):
    try:
        drivers = session.results['Abbreviation'].tolist()
        telemetry_data = []
        for driver in drivers:
            fast_lap = session.laps.pick_driver(driver).pick_fastest()
            if not fast_lap.empty:
                telemetry = fast_lap.get_telemetry()[['X', 'Y', 'Time']].dropna()
                telemetry['Driver'] = driver
                telemetry_data.append(telemetry)
        
        if not telemetry_data:
            logger.warning("No telemetry data for track segment dominance")
            return get_empty_plot("Track Segment Dominance")
        
        all_telemetry = pd.concat(telemetry_data)
        all_telemetry['Time'] = all_telemetry['Time'].dt.total_seconds()
        dominance = all_telemetry.groupby(['X', 'Y'])['Time'].idxmin()
        dominance_data = all_telemetry.loc[dominance]['Driver'].value_counts(normalize=True) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dominance_data.index,
            y=dominance_data.values,
            name="Track Segments"
        ))
        
        fig.update_layout(
            title="Track Segment Dominance",
            xaxis_title="Driver",
            yaxis_title="Percentage of Fastest Segments (%)",
            template="plotly_dark"
        )
        logger.info("Generated track segment dominance plot")
        return pio.to_json(fig)
    except Exception as e:
        logger.error(f"Error in get_track_segment_dominance: {str(e)}")
        return get_empty_plot("Track Segment Dominance")

def get_driver_stats(session):
    try:
        results = session.results
        laps = session.laps
        stats = []
        for driver in results['Abbreviation']:
            driver_laps = laps.pick_driver(driver)
            fastest_lap = driver_laps.pick_fastest()
            avg_speed = driver_laps.get_car_data()['Speed'].mean() if not driver_laps.empty else 0
            stats.append({
                'DriverNumber': results[results['Abbreviation'] == driver]['DriverNumber'].iloc[0],
                'Abbreviation': driver,
                'TeamName': results[results['Abbreviation'] == driver]['TeamName'].iloc[0],
                'Position': str(results[results['Abbreviation'] == driver]['Position'].iloc[0] or 'DNF'),
                'Points': results[results['Abbreviation'] == driver]['Points'].iloc[0],
                'FastestLap': fastest_lap['LapTime'].total_seconds() if not fastest_lap.empty else np.nan,
                'TotalLaps': len(driver_laps),
                'AvgSpeed': avg_speed
            })
        logger.info("Generated driver stats")
        return pd.DataFrame(stats).to_dict('records')
    except Exception as e:
        logger.error(f"Error in get_driver_stats: {str(e)}")
        return []

def get_sector_stats(session):
    try:
        laps = session.laps
        stats = []
        for driver in laps['Driver'].unique():
            driver_laps = laps.pick_driver(driver)
            sector1 = driver_laps['Sector1Time'].dropna().dt.total_seconds()
            sector2 = driver_laps['Sector2Time'].dropna().dt.total_seconds()
            sector3 = driver_laps['Sector3Time'].dropna().dt.total_seconds()
            if not sector1.empty:
                stats.append({
                    'Driver': driver,
                    'Sector1_Mean': sector1.mean(),
                    'Sector1_Std': sector1.std(),
                    'Sector1_CV': sector1.std() / sector1.mean() if sector1.mean() != 0 else 0,
                    'Sector1_Best': sector1.min() if not sector1.empty else np.nan,
                    'Sector2_Mean': sector2.mean(),
                    'Sector2_Std': sector2.std(),
                    'Sector2_CV': sector2.std() / sector2.mean() if sector2.mean() != 0 else 0,
                    'Sector2_Best': sector2.min() if not sector2.empty else np.nan,
                    'Sector3_Mean': sector3.mean(),
                    'Sector3_Std': sector3.std(),
                    'Sector3_CV': sector3.std() / sector3.mean() if sector3.mean() != 0 else 0,
                    'Sector3_Best': sector3.min() if not sector3.empty else np.nan
                })
        logger.info("Generated sector stats")
        return pd.DataFrame(stats).to_dict('records')
    except Exception as e:
        logger.error(f"Error in get_sector_stats: {str(e)}")
        return []

def get_advanced_stats(session):
    try:
        laps = session.laps
        results = session.results
        stats = []
        for driver in results['Abbreviation']:
            driver_laps = laps.pick_driver(driver)
            fastest_lap = driver_laps.pick_fastest()
            lap_times = driver_laps['LapTime'].dropna().dt.total_seconds()
            sector1 = driver_laps['Sector1Time'].dropna().dt.total_seconds()
            sector2 = driver_laps['Sector2Time'].dropna().dt.total_seconds()
            sector3 = driver_laps['Sector3Time'].dropna().dt.total_seconds()
            car_data = driver_laps.get_car_data().mean() if not driver_laps.empty else pd.Series()
            
            # Calculate DPI components
            fastest_lap_time = fastest_lap['LapTime'].total_seconds() if not fastest_lap.empty else np.inf
            lap_time_norm = 1 / fastest_lap_time if fastest_lap_time != np.inf else 0
            sector_cv = (sector1.std() / sector1.mean() + sector2.std() / sector2.mean() + sector3.std() / sector3.mean()) / 3 if all([sector1.mean(), sector2.mean(), sector3.mean()]) else 0
            avg_speed = car_data.get('Speed', 0)
            throttle_eff = car_data.get('Throttle', 0) / avg_speed if avg_speed != 0 else 0
            
            # DPI: Weighted sum (normalized)
            dpi = (0.4 * lap_time_norm / max(lap_times.min(), 1e-6) + 
                   0.3 * (1 - sector_cv) + 
                   0.2 * avg_speed / max(lap_times.mean(), 1e-6) + 
                   0.1 * throttle_eff) * 100
            
            # Lap time trend slope
            slope = 0
            if len(lap_times) > 1:
                slope, _, _, _, _ = linregress(driver_laps['LapNumber'], lap_times)
            
            stats.append({
                'Driver': driver,
                'DPI': dpi,
                'FastestLap': fastest_lap_time if fastest_lap_time != np.inf else np.nan,
                'SectorCV': sector_cv,
                'AvgSpeed': avg_speed,
                'ThrottleEfficiency': throttle_eff,
                'LapTimeTrendSlope': slope
            })
        logger.info("Generated advanced stats")
        return pd.DataFrame(stats).to_dict('records')
    except Exception as e:
        logger.error(f"Error in get_advanced_stats: {str(e)}")
        return []

def get_telemetry_stats(session, drivers):
    try:
        stats = []
        for driver in drivers:
            fast_lap = session.laps.pick_driver(driver).pick_fastest()
            if fast_lap.empty:
                continue
            telemetry = fast_lap.get_telemetry()
            throttle_ratio = telemetry['Throttle'].mean() / telemetry['Speed'].mean() if telemetry['Speed'].mean() != 0 else 0
            brake_freq = (telemetry['Brake'] > 50).mean() if 'Brake' in telemetry else 0
            stats.append({
                'Driver': driver,
                'ThrottleToSpeedRatio': throttle_ratio,
                'BrakeFrequency': brake_freq,
                'MaxSpeed': telemetry['Speed'].max() if not telemetry.empty else 0
            })
        logger.info("Generated telemetry stats")
        return pd.DataFrame(stats).to_dict('records')
    except Exception as e:
        logger.error(f"Error in get_telemetry_stats: {str(e)}")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    year = int(request.form.get('year', 2025))
    grand_prix = request.form.get('grand_prix', 'Austria')
    drivers = request.form.getlist('drivers', None) or []
    tire_driver = request.form.get('tire_driver', None)
    heatmap_driver = request.form.get('heatmap_driver', None)
    gear_driver = request.form.get('gear_driver', None)
    
    session, warning = load_session(year, grand_prix)
    if session is None:
        logger.error(f"Rendering error page for {year} {grand_prix}")
        return render_template(
            'error.html',
            error_message=warning,
            years=YEARS,
            grand_prix_list=GRAND_PRIX_LIST,
            year=year
        )
    
    available_drivers = sorted(session.results['Abbreviation'].tolist())
    drivers = [d for d in drivers if d in available_drivers][:2] or available_drivers[:2]
    tire_driver = tire_driver if tire_driver in available_drivers else available_drivers[0]
    heatmap_driver = heatmap_driver if heatmap_driver in available_drivers else available_drivers[0]
    gear_driver = gear_driver if gear_driver in available_drivers else available_drivers[0]
    
    driver_stats = get_driver_stats(session)
    sector_stats = get_sector_stats(session)
    
    logger.info(f"Rendering dashboard for {year} {grand_prix}")
    return render_template(
        'index.html',
        years=YEARS,
        grand_prix_list=GRAND_PRIX_LIST,
        year=year,
        grand_prix=grand_prix,
        drivers=available_drivers,
        selected_drivers=drivers,
        tire_driver=tire_driver,
        heatmap_driver=heatmap_driver,
        gear_driver=gear_driver,
        driver_stats=driver_stats,
        sector_stats=sector_stats,
        warning=warning
    )

@app.route('/stats', methods=['GET', 'POST'])
def stats():
    year = int(request.form.get('year', 2025))
    grand_prix = request.form.get('grand_prix', 'Austria')
    drivers = request.form.getlist('drivers', None) or []
    
    session, warning = load_session(year, grand_prix)
    if session is None:
        logger.error(f"Rendering error page for {year} {grand_prix}")
        return render_template(
            'error.html',
            error_message=warning,
            years=YEARS,
            grand_prix_list=GRAND_PRIX_LIST,
            year=year
        )
    
    available_drivers = sorted(session.results['Abbreviation'].tolist())
    drivers = [d for d in drivers if d in available_drivers][:5] or available_drivers[:5]
    
    advanced_stats = get_advanced_stats(session)
    
    logger.info(f"Rendering stats page for {year} {grand_prix}")
    return render_template(
        'stats.html',
        years=YEARS,
        grand_prix_list=GRAND_PRIX_LIST,
        year=year,
        grand_prix=grand_prix,
        drivers=available_drivers,
        selected_drivers=drivers,
        advanced_stats=advanced_stats,
        warning=warning
    )

@app.route('/telemetry', methods=['GET', 'POST'])
def telemetry():
    year = int(request.form.get('year', 2025))
    grand_prix = request.form.get('grand_prix', 'Austria')
    telemetry_driver = request.form.get('telemetry_driver', None)
    
    session, warning = load_session(year, grand_prix)
    if session is None:
        logger.error(f"Rendering error page for {year} {grand_prix}")
        return render_template(
            'error.html',
            error_message=warning,
            years=YEARS,
            grand_prix_list=GRAND_PRIX_LIST,
            year=year
        )
    
    available_drivers = sorted(session.results['Abbreviation'].tolist())
    telemetry_driver = telemetry_driver if telemetry_driver in available_drivers else available_drivers[0]
    
    telemetry_stats = get_telemetry_stats(session, available_drivers)
    
    logger.info(f"Rendering telemetry page for {year} {grand_prix}")
    return render_template(
        'telemetry.html',
        years=YEARS,
        grand_prix_list=GRAND_PRIX_LIST,
        year=year,
        grand_prix=grand_prix,
        drivers=available_drivers,
        telemetry_driver=telemetry_driver,
        telemetry_stats=telemetry_stats,
        warning=warning
    )

@app.route('/api/plot/<plot_type>', methods=['GET'])
def get_plot(plot_type):
    year = int(request.args.get('year', 2025))
    grand_prix = request.args.get('grand_prix', 'Austria')
    drivers = request.args.getlist('drivers') or []
    tire_driver = request.args.get('tire_driver', None)
    heatmap_driver = request.args.get('heatmap_driver', None)
    gear_driver = request.args.get('gear_driver', None)
    telemetry_driver = request.args.get('telemetry_driver', None)
    metric = request.args.get('metric', 'Speed')
    
    session, _ = load_session(year, grand_prix)
    if session is None:
        return jsonify({'error': f"Could not load data for {grand_prix} in {year}"}), 500
    
    available_drivers = sorted(session.results['Abbreviation'].tolist())
    drivers = [d for d in drivers if d in available_drivers][:5] or available_drivers[:2]
    tire_driver = tire_driver if tire_driver in available_drivers else available_drivers[0]
    heatmap_driver = heatmap_driver if heatmap_driver in available_drivers else available_drivers[0]
    gear_driver = gear_driver if gear_driver in available_drivers else available_drivers[0]
    telemetry_driver = telemetry_driver if telemetry_driver in available_drivers else available_drivers[0]
    
    try:
        if plot_type == 'speed':
            plot_json = get_driver_comparison_plot(session, drivers, metric=metric)
        elif plot_type == 'sector':
            plot_json = get_sector_time_plot(session, drivers)
        elif plot_type == 'tire':
            plot_json = get_tire_strategy_plot(session, tire_driver)
        elif plot_type == 'lap_time':
            plot_json = get_lap_time_distribution(session)
        elif plot_type == 'gear':
            plot_json = get_gear_shift_plot(session, gear_driver)
        elif plot_type == 'heatmap':
            plot_json = get_telemetry_heatmap(session, heatmap_driver, metric=metric)
        elif plot_type == 'dominance':
            plot_json = get_track_dominance_map(session)
        elif plot_type == 'lap_trend':
            plot_json = get_lap_time_trend(session, drivers)
        elif plot_type == 'consistency':
            plot_json = get_driver_consistency_plot(session)
        elif plot_type == 'sector_dominance':
            plot_json = get_sector_dominance_plot(session)
        elif plot_type == 'track_segment':
            plot_json = get_track_segment_dominance(session)
        else:
            return jsonify({'error': f"Invalid plot type: {plot_type}"}), 400
        
        return jsonify({'data': plot_json})
    except Exception as e:
        logger.error(f"Error generating {plot_type} plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<stat_type>', methods=['GET'])
@app.route('/api/export/<stat_type>', methods=['GET'])
def export_stats(stat_type):
    year = int(request.args.get('year', 2025))
    grand_prix = request.args.get('grand_prix', 'Austria')
    
    session, _ = load_session(year, grand_prix)
    if session is None:
        return jsonify({'error': f"Could not load data for {grand_prix} in {year}"}), 500
    
    try:
        if stat_type == 'driver_stats':
            data = pd.DataFrame(get_driver_stats(session))
        elif stat_type == 'sector_stats':
            data = pd.DataFrame(get_sector_stats(session))
        elif stat_type == 'advanced_stats':
            data = pd.DataFrame(get_advanced_stats(session))
        elif stat_type == 'telemetry_stats':
            data = pd.DataFrame(get_telemetry_stats(session, session.results['Abbreviation'].tolist()))
        else:
            return jsonify({'error': f"Invalid stat type: {stat_type}"}), 400
        
        # Create BytesIO buffer instead of StringIO
        csv_buffer = BytesIO()
        data.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{stat_type}_{year}_{grand_prix}.csv"
        )
    except Exception as e:
        logger.error(f"Error exporting {stat_type}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)