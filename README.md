# F1 Stats Dashboard & API

This project is a web-based dashboard and API for exploring and analyzing Formula 1 race data using [FastF1](https://theoehrly.github.io/Fast-F1/), [Flask](https://flask.palletsprojects.com/), and [Plotly](https://plotly.com/python/). It provides interactive visualizations, advanced statistics, and data export features for F1 qualifying sessions across multiple years and Grand Prix events.

## Features

- **Interactive Dashboard**: Visualize and compare driver performance, sector times, tire strategies, lap time trends, and more.
- **Advanced Analytics**: Includes metrics like Driver Performance Index (DPI), lap time trend slopes, sector consistency, and telemetry efficiency.
- **Multiple Years & Events**: Supports data for 2023, 2024, and 2025 (with fallback to 2023 if newer data is unavailable).
- **Export to CSV**: Download driver, sector, advanced, and telemetry statistics as CSV files for further analysis.
- **API Endpoints**: Fetch plot data and statistics programmatically via RESTful endpoints.
- **Caching**: Uses FastF1's cache to speed up repeated data access.

## Project Structure

```
F1 Stats/
│   main.py
│
├── cache_dir/                # FastF1 cache (auto-generated)
│   └── ...
├── templates/                # HTML templates for Flask
│   ├── error.html
│   ├── index.html
│   ├── stats.html
│   └── telemetry.html
```

## Requirements

- Python 3.8+
- pip

### Python Packages
- fastf1
- flask
- pandas
- numpy
- plotly
- scipy

Install all dependencies with:

```bash
pip install fastf1 flask pandas numpy plotly scipy
```

## Usage

1. **Start the Server**

   ```bash
   python main.py
   ```

2. **Open in Browser**

   Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the dashboard.

3. **Explore**
   - Select year and Grand Prix.
   - Choose drivers for comparison.
   - View stats, telemetry, and plots.
   - Download CSVs from the stats and telemetry pages.

## API Endpoints

### Plots
- `/api/plot/<plot_type>`: Returns Plotly JSON for the requested plot type.
  - Query params: `year`, `grand_prix`, `drivers`, `tire_driver`, `heatmap_driver`, `gear_driver`, `telemetry_driver`, `metric`
  - Plot types: `speed`, `sector`, `tire`, `lap_time`, `gear`, `heatmap`, `dominance`, `lap_trend`, `consistency`, `sector_dominance`, `track_segment`

### Data Export
- `/api/export/<stat_type>`: Download CSV for the requested stat type.
  - Query params: `year`, `grand_prix`
  - Stat types: `driver_stats`, `sector_stats`, `advanced_stats`, `telemetry_stats`

## Customization
- Add or modify HTML templates in `templates/` for UI changes.
- Extend or adjust analytics in `main.py` for new metrics or plots.

## Notes
- The app will fallback to 2023 data if newer data is unavailable for a session.
- The FastF1 cache is stored in `cache_dir/` and will grow as you explore more sessions.

## License
This project is for educational and personal use. See [FastF1 license](https://github.com/theOehrly/Fast-F1/blob/main/LICENSE) for data usage terms.

---
