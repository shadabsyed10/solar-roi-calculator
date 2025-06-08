# ☀️ Solar ROI Calculator

A comprehensive Streamlit web application for calculating Return on Investment (ROI) for solar panel installations with advanced financial modeling and shading analysis capabilities.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### ✅ Currently Available
- **Comprehensive ROI Analysis**: Calculate NPV, IRR, payback period, and LCOE
- **Location-based Solar Calculations**: Uses pvlib for accurate solar irradiance data
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Battery Storage Analysis**: Include battery systems in your calculations
- **Net Metering Support**: Factor in energy export rates
- **Financial Modeling**: 
  - Tax benefits and subsidies
  - Maintenance costs
  - Inverter replacement scheduling
  - Annual electricity rate increases
- **Multiple Roof Shapes**: Support for flat, gabled, and hipped roofs
- **Export Capabilities**: Download results as PDF reports

### 🚧 Work in Progress
- **3D Shading Analysis**: Blender API integration for advanced shading calculations
- **Obstacle Modeling**: 3D visualization of nearby buildings and trees
- **Advanced Shading Reports**: Detailed monthly/hourly shading impact analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Blender (optional, for future shading analysis features)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/shadabsyed10/solar-roi-calculator.git
cd solar-roi-calculator

2. Create a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

Running the Application
1. Start the Streamlit server:
streamlit run calculate_roi.py

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Enter your parameters and calculate your solar ROI!


📊 How to Use
Step 1: Location & System Details
Enter your latitude and longitude or use the location finder
Specify roof area in square meters
Choose roof shape (flat, gabled, or hipped)
Set panel specifications (efficiency, tilt, azimuth)
Step 2: Financial Parameters
Installation cost per kW
Electricity rates and annual increases
Subsidies and tax benefits
Maintenance costs as percentage of system cost
System lifetime (typically 25 years)
Step 3: Optional Features
Battery storage capacity and costs
Net metering export rates
Inverter replacement scheduling
Step 4: View Results
Financial metrics: NPV, IRR, payback period
Energy production: Monthly and annual estimates
Cost savings: Lifetime electricity cost savings
Environmental impact: CO2 emissions avoided
📋 Input Parameters Guide
Technical Parameters
Parameter	Description	Typical Range
Panel Efficiency	Solar panel efficiency percentage	15-22%
System Losses	Total system losses	10-20%
Panel Tilt	Angle from horizontal	0-60°
Panel Azimuth	Direction (0=North, 180=South)	120-240°
Panel Degradation	Annual efficiency loss	0.3-0.8%
Financial Parameters
Parameter	Description	Notes
Installation Cost	Cost per kW installed	Includes panels, inverters, installation
Electricity Rate	Current rate per kWh	Check your utility bill
Annual Rate Increase	Expected yearly rate increase	Historical average: 2-4%
Discount Rate	Your cost of capital	Typical: 3-8%
🔧 Configuration
Environment Variables (Optional)
Create a .env file in the project root:

env


BLENDER_PATH=/path/to/blender/executable
DEFAULT_LOCATION_LAT=28.6139
DEFAULT_LOCATION_LON=77.2090
Customizing Calculations
You can modify calculation parameters in the calculate_roi.py file:

Default system lifetime
Standard panel specifications
Regional electricity rate assumptions
⚠️ Known Limitations
Blender Shading Analysis (WIP)
The 3D shading analysis feature using Blender is currently under development:

Status: Not functional in current GUI
Expected: Future release will include full 3D shading simulation
Workaround: Use simplified shading estimates in current version
Current Shading Handling
Basic obstacle input (distance, height, azimuth)
Simplified shading calculations
Conservative estimates for unknown shading factors
🛠️ Development
Project Structure
solar-roi-calculator/
├── calculate_roi.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
└── temp_blender/            # Temporary Blender files (auto-created)
Contributing
Fork the repository
Create a feature branch: git checkout -b feature/amazing-feature
Commit changes: git commit -m 'Add amazing feature'
Push to branch: git push origin feature/amazing-feature
Open a Pull Request
Running Tests
bash


# Tests will be added in future versions
python -m pytest tests/
📚 Dependencies
Core Libraries
streamlit: Web application framework
pandas: Data manipulation and analysis
numpy: Numerical computing
pvlib: Solar energy calculations
plotly: Interactive visualizations
geopy: Geocoding services
Optional Dependencies
blender: 3D modeling and shading analysis (WIP)
🐛 Troubleshooting
Common Issues
1. Import Errors

bash


# Ensure all dependencies are installed
pip install -r requirements.txt
2. Location Services

Check internet connection for geocoding
Manually enter coordinates if automatic location fails
3. Calculation Errors

Verify all input parameters are within reasonable ranges
Check for negative values where not applicable
4. Performance Issues

Large roof areas may slow calculations
Consider reducing analysis granularity for faster results
📈 Roadmap
Version 2.0 (Planned)
✅ Complete Blender integration for shading analysis
✅ Advanced 3D visualization
✅ Weather data integration
✅ Multiple scenario comparison
✅ API endpoints for external integration
Version 2.1 (Future)
✅ Machine learning for optimal panel placement
✅ Real-time energy monitoring integration
✅ Mobile-responsive design
✅ Multi-language support
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
pvlib-python for solar position and irradiance calculations
Streamlit for the amazing web app framework
Plotly for interactive visualizations
Blender for 3D modeling capabilities (WIP)
📞 Support
Issues: GitHub Issues
Discussions: GitHub Discussions
Email: shadabsyed07@gmail.com
🌍 Contributing to Clean Energy
This tool helps make solar energy more accessible by providing transparent, detailed financial analysis. Every solar installation contributes to a cleaner, more sustainable future.

⭐ If this project helps you, please consider giving it a star on GitHub!

Last updated: June 2025


This README provides:
- Clear installation and usage instructions
- Honest disclosure about WIP features
- Comprehensive parameter guides
- Troubleshooting section
- Professional presentation with badges and emojis
- Future roadmap to show project direction

You can customize the email, add screenshots, or modify any sections as needed!
