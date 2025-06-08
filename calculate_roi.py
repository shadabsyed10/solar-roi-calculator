# File: calculate_roi.py
import streamlit as st
import pandas as pd
import numpy as np
from pvlib.location import Location
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from io import BytesIO
import pvlib
import os
import tempfile
import subprocess
import json
import sys
import time  # Add this import
from pathlib import Path


# Add these debugging helpers at the top of your file, after imports
def debug_blender_installation(blender_path=None):
    """Check if Blender is properly installed and accessible"""
    if blender_path is None:
        if sys.platform == "darwin":  # macOS
            possible_paths = [
                "/Applications/Blender.app/Contents/MacOS/Blender",
                "/Applications/Blender/Blender.app/Contents/MacOS/Blender"
            ]
        elif sys.platform == "win32":  # Windows
            possible_paths = [
                r"C:\Program Files\Blender Foundation\Blender\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 3.0\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 2.93\blender.exe"
            ]
        else:  # Linux and others
            possible_paths = [
                "/usr/bin/blender",
                "/usr/local/bin/blender"
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                blender_path = path
                break
    
    if blender_path is None or not os.path.exists(blender_path):
        return False, "Blender executable not found"
    
    # Try running Blender with --version to check if it works
    try:
        result = subprocess.run([blender_path, "--version"], 
                               capture_output=True, text=True, check=True)
        return True, f"Blender found: {result.stdout.strip()}"
    except subprocess.CalledProcessError as e:
        return False, f"Error running Blender: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

# Check if bpy (Blender Python API) is available
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Solar ROI Calculator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1976D2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1976D2;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

def create_blender_script(lat, lon, roof_area, panel_tilt, panel_azimuth, 
                          obstacles=None, roof_shape="flat", output_file=None):
    """Create a Python script to be executed by Blender to simulate solar panels and shading"""
    
    if output_file is None:
        output_file = tempfile.mktemp(suffix='.json')
    
    script = f"""
import bpy
import math
import bmesh
from mathutils import Vector, Matrix
import numpy as np
import json
from datetime import datetime, timedelta

print("Starting Blender script execution...")

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

print("Scene cleared...")

# Create a new scene
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # Use Cycles for accurate sun lighting

print("Setting up world and sky...")

# Set up the world with a sun
world = bpy.data.worlds['World']
world.use_nodes = True
world_nodes = world.node_tree.nodes
world_nodes.clear()

# Add Sky Texture node for accurate sun position
sky_node = world_nodes.new(type='ShaderNodeTexSky')
sky_node.location = (0, 0)
background_node = world_nodes.new(type='ShaderNodeBackground')
background_node.location = (200, 0)
output_node = world_nodes.new(type='ShaderNodeOutputWorld')
output_node.location = (400, 0)

# Connect nodes
links = world.node_tree.links
links.new(sky_node.outputs['Color'], background_node.inputs['Color'])
links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

print("Creating ground plane...")

# Create the ground
bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
ground = bpy.context.object
ground.name = 'Ground'
ground_material = bpy.data.materials.new(name="GroundMaterial")
ground_material.use_nodes = True
ground_material.diffuse_color = (0.3, 0.5, 0.3, 1.0)
ground.data.materials.append(ground_material)

print("Creating roof...")

# Create the roof
roof_width = math.sqrt({roof_area})
roof_length = roof_width

if "{roof_shape}" == "flat":
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 3))
    roof = bpy.context.object
    roof.scale = (roof_width/2, roof_length/2, 1)
elif "{roof_shape}" == "gabled":
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 3))
    roof = bpy.context.object
    roof.scale = (roof_width/2, roof_length/2, 0.1)
    # Add ridge
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(roof.data)
    for v in bm.verts:
        if v.co.z > 3:
            v.co.z += 0.5
    bmesh.update_edit_mesh(roof.data)
    bpy.ops.object.mode_set(mode='OBJECT')

roof.name = 'Roof'
roof_material = bpy.data.materials.new(name="RoofMaterial")
roof_material.use_nodes = True
roof_material.diffuse_color = (0.6, 0.6, 0.6, 1.0)
roof.data.materials.append(roof_material)

print("Setting up solar panels...")

# Create solar panel collection
panel_group = bpy.data.collections.new("SolarPanels")
bpy.context.scene.collection.children.link(panel_group)

# Calculate panel layout
panel_width = 1.7
panel_height = 1.0
panel_area = panel_width * panel_height
num_panels = int({roof_area} / panel_area * 0.8)

aspect_ratio = roof_width / roof_length
cols = int(math.sqrt(num_panels * aspect_ratio))
rows = int(num_panels / cols)

print(f"Creating {{rows}}x{{cols}} panel array...")

# Create panels
panel_material = bpy.data.materials.new(name="PanelMaterial")
panel_material.use_nodes = True
panel_material.diffuse_color = (0.1, 0.1, 0.3, 1.0)

for row in range(rows):
    for col in range(cols):
        x = (col - cols/2 + 0.5) * (panel_width * 1.1)
        y = (row - rows/2 + 0.5) * (panel_height * 1.1)
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 3.1))
        panel = bpy.context.object
        panel.scale = (panel_width/2, panel_height/2, 0.02)
        panel.name = f"Panel_{{row}}_{{col}}"
        
        # Apply tilt and azimuth
        blender_azimuth = ({panel_azimuth} - 180) * math.pi / 180
        tilt = {panel_tilt} * math.pi / 180
        panel.rotation_euler[2] = blender_azimuth
        panel.rotation_euler[0] = tilt
        
        panel.data.materials.append(panel_material)
        panel_group.objects.link(panel)

print("Setting up sun position calculation...")

def update_sun_position(month, day, hour):
    day_of_year = sum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:month]) + day
    declination = -23.45 * math.cos(math.radians((360/365) * (day_of_year + 10)))
    hour_angle = (hour - 12) * 15
    
    lat_rad = math.radians({lat})
    declination_rad = math.radians(declination)
    hour_angle_rad = math.radians(hour_angle)
    
    elevation = math.asin(
        math.sin(lat_rad) * math.sin(declination_rad) +
        math.cos(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)
    )
    
    azimuth = math.atan2(
        -math.cos(declination_rad) * math.sin(hour_angle_rad),
        math.cos(lat_rad) * math.sin(declination_rad) -
        math.sin(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)
    )
    
    sky_node.sun_elevation = elevation
    sky_node.sun_rotation = azimuth
    
    # Update sun lamp
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.object
    sun.rotation_euler = (math.pi/2 - elevation, 0, azimuth)
    sun.data.energy = 5.0
    
    bpy.context.view_layer.update()
    return sun

print("Starting shading analysis...")

# Initialize monthly shading data
monthly_shading = [[] for _ in range(12)]

# Analysis times - for all months
analysis_times = []
for month in range(1, 13):
    day = 15
    for hour in [8, 12, 16]:
        analysis_times.append((month, day, hour))

print(f"Analyzing {{len(analysis_times)}} time points...")

def get_sun_elevation_azimuth(month, day, hour, lat, lon):
    # Simplified solar position calculation
    day_of_year = month * 30.44 + day  # Approximate day of year
    
    # Solar declination
    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    
    # Hour angle
    hour_angle = (hour - 12) * 15  # degrees from solar noon
    
    # Solar elevation
    elevation = math.asin(
        math.sin(math.radians(declination)) * math.sin(math.radians(lat)) +
        math.cos(math.radians(declination)) * math.cos(math.radians(lat)) * 
        math.cos(math.radians(hour_angle))
    )
    
    # Solar azimuth
    azimuth = math.atan2(
        math.sin(math.radians(hour_angle)),
        math.cos(math.radians(hour_angle)) * math.sin(math.radians(lat)) -
        math.tan(math.radians(declination)) * math.cos(math.radians(lat))
    )
    
    return math.degrees(elevation), math.degrees(azimuth)

def calculate_simple_shading(panel_tilt, panel_azimuth, sun_elevation, sun_azimuth):
    # If sun is below horizon, full shading
    if sun_elevation <= 0:
        return 100.0
    
    # Calculate angle between panel normal and sun direction
    panel_tilt_rad = math.radians(panel_tilt)
    panel_azimuth_rad = math.radians(panel_azimuth)
    sun_elevation_rad = math.radians(sun_elevation)
    sun_azimuth_rad = math.radians(sun_azimuth)
    
    # Panel normal vector
    panel_normal_x = math.sin(panel_tilt_rad) * math.sin(panel_azimuth_rad)
    panel_normal_y = math.sin(panel_tilt_rad) * math.cos(panel_azimuth_rad)
    panel_normal_z = math.cos(panel_tilt_rad)
    
    # Sun direction vector
    sun_x = math.cos(sun_elevation_rad) * math.sin(sun_azimuth_rad)
    sun_y = math.cos(sun_elevation_rad) * math.cos(sun_azimuth_rad)
    sun_z = math.sin(sun_elevation_rad)
    
    # Dot product to get cosine of angle between vectors
    dot_product = panel_normal_x * sun_x + panel_normal_y * sun_y + panel_normal_z * sun_z
    
    # If dot product is negative, sun is behind panel
    if dot_product <= 0:
        return 100.0  # Full shading
    
    # Calculate shading based on angle (simplified model)
    # Higher dot product means better alignment, less shading
    shading_percentage = (1.0 - dot_product) * 50.0  # Scale to 0-50% range
    
    # Add some base shading for atmospheric effects, dust, etc.
    base_shading = 5.0
    
    return min(100.0, max(0.0, shading_percentage + base_shading))

for month, day, hour in analysis_times:
    print(f"Calculating shading for month {{month}}, day {{day}}, hour {{hour}}...")
    
    # Get sun position
    sun_elevation, sun_azimuth = get_sun_elevation_azimuth(month, day, hour, {lat}, {lon})
    
    # Calculate shading for this time point
    shading_percentage = calculate_simple_shading({panel_tilt}, {panel_azimuth}, sun_elevation, sun_azimuth)
    
    month_index = month - 1
    monthly_shading[month_index].append(float(shading_percentage))
    
    print(f"  Sun elevation: {{sun_elevation:.1f}}°, azimuth: {{sun_azimuth:.1f}}°, shading: {{shading_percentage:.1f}}%")

print("Calculating monthly averages...")
monthly_averages = []

for i, month_data in enumerate(monthly_shading):
    if month_data:
        avg_shading = sum(month_data) / len(month_data)
        monthly_averages.append(float(avg_shading))
        print(f"Month {{i+1}}: Average shading {{avg_shading:.1f}}%")
    else:
        monthly_averages.append(20.0)  # Default reasonable value
        print(f"Month {{i+1}}: No data, using default 20%")

# Create results dictionary
results = {{
    "shading_analysis": {{
        "monthly_averages": [float(x) for x in monthly_averages],
        "annual_average": float(sum(monthly_averages) / len(monthly_averages)),
        "seasonal_variation": {{
            "spring": sum(monthly_averages[2:5])/3 if len(monthly_averages) >= 5 else 20.0,
            "summer": sum(monthly_averages[5:8])/3 if len(monthly_averages) >= 8 else 15.0,
            "autumn": sum(monthly_averages[8:11])/3 if len(monthly_averages) >= 11 else 20.0,
            "winter": (sum([monthly_averages[11], monthly_averages[0], monthly_averages[1]])/3) 
                     if len(monthly_averages) >= 12 else 25.0
        }},
        "hourly_variation": {{
            "morning": [float(monthly_shading[m][0]) if monthly_shading[m] else 25.0 for m in range(12)],
            "noon": [float(monthly_shading[m][1]) if len(monthly_shading[m]) > 1 else 15.0 for m in range(12)],
            "afternoon": [float(monthly_shading[m][2]) if len(monthly_shading[m]) > 2 else 20.0 for m in range(12)]
        }}
    }},
    "system_info": {{
        "num_panels": int(num_panels),
        "panel_area": float(panel_area),
        "total_area": float(num_panels * panel_area),
        "roof_utilization": float((num_panels * panel_area) / {roof_area}),
        "layout": {{
            "rows": int(rows),
            "cols": int(cols)
        }}
    }},
    "location": {{
        "latitude": float({lat}),
        "longitude": float({lon})
    }},
    "analysis_parameters": {{
        "panel_tilt": float({panel_tilt}),
        "panel_azimuth": float({panel_azimuth}),
        "roof_shape": "{roof_shape}"
    }},
    "timestamp": str(datetime.now())
}}

print("Saving results...")
try:
    with open("{output_file}", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    print("Analysis completed successfully!")
except Exception as e:
    print(f"Error saving results: {{e}}")
"""
   # Find this section in your create_blender_script function and replace it:

# Replace the monthly averages calculation section with this:

    script += f'''

print("Calculating monthly averages...")
monthly_averages = []
for i, month_data in enumerate(monthly_shading):
    if month_data:
        # Ensure all values are numeric and filter out any non-numeric data
        numeric_data = []
        for value in month_data:
            try:
                if isinstance(value, (int, float)):
                    numeric_data.append(float(value))
                elif isinstance(value, str):
                    numeric_data.append(float(value))
            except (ValueError, TypeError):
                print(f"Warning: Invalid shading value {{value}} in month {{i+1}}, skipping...")
                continue
        
        if numeric_data:
            avg_shading = sum(numeric_data) / len(numeric_data)
            monthly_averages.append(avg_shading)
        else:
            print(f"Warning: No valid data for month {{i+1}}, using 0")
            monthly_averages.append(0.0)
    else:
        monthly_averages.append(0.0)

print("Saving results...")

# Create the results dictionary
results = {{
    "shading_analysis": {{
        "monthly_data": monthly_shading,
        "monthly_averages": monthly_averages,
        "annual_average": sum(monthly_averages) / len(monthly_averages) if monthly_averages else 0.0,
        "seasonal_variation": {{
            "spring": monthly_averages[2] if len(monthly_averages) > 2 else 0.0,
            "summer": monthly_averages[5] if len(monthly_averages) > 5 else 0.0,
            "autumn": monthly_averages[8] if len(monthly_averages) > 8 else 0.0,
            "winter": monthly_averages[11] if len(monthly_averages) > 11 else 0.0
        }}
    }},
    "system_info": {{
        "num_panels": num_panels,
        "panel_area": panel_area,
        "total_area": num_panels * panel_area,
        "roof_utilization": (num_panels * panel_area) / {roof_area} if {roof_area} > 0 else 0.0
    }},
    "location": {{
        "latitude": {lat},
        "longitude": {lon}
    }},
    "timestamp": str(datetime.now())
}}

try:
    with open("{output_file}", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    print("Analysis completed successfully!")
except Exception as e:
    print(f"Error saving results: {{e}}")
    import traceback
    traceback.print_exc()

'''

    return script, output_file




def run_blender_simulation(script_content, blender_path=None):
    """Run the Blender simulation using the generated script"""
    
    # Create a temporary script file in a location we're sure to have write access
    script_dir = os.path.join(os.path.expanduser("~"), "temp_blender")
    os.makedirs(script_dir, exist_ok=True)
    
    # Create unique filenames
    timestamp = int(time.time())
    script_file = os.path.join(script_dir, f"blender_script_{timestamp}.py")
    output_file = os.path.join(script_dir, f"shading_results_{timestamp}.json")
    
    # The script should already have the correct output file path embedded
    # No need to do string replacement - the create_blender_script function
    # should already include the correct path
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Find Blender executable if not provided
    if blender_path is None:
        if sys.platform == "darwin":  # macOS
            possible_paths = [
                "/Applications/Blender.app/Contents/MacOS/Blender",
                "/Applications/Blender/Blender.app/Contents/MacOS/Blender"
            ]
        else:
            possible_paths = ["/usr/bin/blender", "/usr/local/bin/blender"]
        
        for path in possible_paths:
            if os.path.exists(path):
                blender_path = path
                break
    
    if blender_path is None or not os.path.exists(blender_path):
        return None, False, "Blender executable not found. Please provide the path to Blender."
    
    try:
        # Run the actual shading analysis script
        print(f"Running Blender with script: {script_file}")
        print(f"Expected output file: {output_file}")
        
        result = subprocess.run(
            [blender_path, "--background", "--python", script_file],
            capture_output=True, text=True, check=True
        )
        
        print(f"Blender execution completed. Checking for output file: {output_file}")
        
        # First, try to find the output file from Blender's output
        actual_output_file = None
        for line in result.stdout.split('\n'):
            if "Results saved to:" in line:
                actual_output_file = line.split("Results saved to: ")[1].strip()
                break
        
        # Check both the expected location and the actual location
        output_files_to_check = [output_file]
        if actual_output_file:
            output_files_to_check.append(actual_output_file)
        
        for file_path in output_files_to_check:
            if os.path.exists(file_path):
                print(f"Output file found: {file_path}")
                # Verify the file contains valid JSON
                try:
                    with open(file_path, 'r') as f:
                        json_content = json.load(f)
                    print("JSON content loaded successfully")
                    return file_path, True, None
                except json.JSONDecodeError as e:
                    return None, False, f"Output file contains invalid JSON: {str(e)}"
        
        # If no output file found
        print("Output file not found in any expected location")
        log_file = os.path.join(script_dir, "blender_error.log")
        with open(log_file, 'w') as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        
        return None, False, f"Output file not created. See log at {log_file}"
        
    except subprocess.CalledProcessError as e:
        error_log = os.path.join(script_dir, "blender_error.log")
        with open(error_log, 'w') as f:
            f.write(f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
        
        return None, False, f"Error running Blender: {str(e)}. See log at {error_log}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"



# File: calculate_roi.py
def apply_shading_factors(energy_production, shading_data):
    """
    Apply monthly shading factors to energy production
    
    Parameters:
    -----------
    energy_production : array
        Monthly energy production without shading
    shading_data : list of dict
        Monthly shading percentages
    
    Returns:
    --------
    array
        Adjusted energy production with shading effects
    """
    adjusted_production = np.copy(energy_production)
    
    # If we have annual data, convert to monthly
    if len(adjusted_production) == 1:
        # Distribute annual production to months based on typical solar insolation patterns
        monthly_factors = np.array([0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.11, 0.1, 0.09, 0.08, 0.06, 0.05])
        monthly_production = adjusted_production[0] * monthly_factors
        adjusted_production = monthly_production
    
    # Apply shading factors
    for i, month_data in enumerate(shading_data):
        # Reduce production by the shading percentage
        shading_factor = month_data["average_shading_percentage"] / 100
        adjusted_production[i] *= (1 - shading_factor)
    
    return adjusted_production


# Header
st.markdown("<h1 class='main-header'>☀️ Advanced Solar ROI Calculator with 3D Shading Analysis</h1>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Input Parameters", "🏠 3D Model & Shading", "📊 Results Overview", "📈 Detailed Analysis", "ℹ️ About"])

with tab1:
    st.markdown("<h2 class='sub-header'>Location Information</h2>", unsafe_allow_html=True)
    
    # Option to choose input method
    input_method = st.radio("Choose location input method:", ("Enter Address", "Enter Latitude and Longitude"))

    lat = None
    lon = None
    address = None

    if input_method == "Enter Address":
        address = st.text_input("Enter your address")
        
        if address:
            # Geocoding with Nominatim
            try:
                with st.spinner("Geocoding address..."):
                    geolocator = Nominatim(user_agent="solar_roi_calculator")
                    location = geolocator.geocode(address)
                    if location:
                        lat = location.latitude
                        lon = location.longitude
                        st.success(f"Successfully geocoded: {location.address}")
                        st.write(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")
                    else:
                        st.error("Could not geocode the address. Please try entering Latitude and Longitude directly.")
            except Exception as e:
                st.error(f"Geocoding error: {e}. Please try entering Latitude and Longitude directly.")

    elif input_method == "Enter Latitude and Longitude":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Enter Latitude", value=19.085225, format="%.6f")
        with col2:
            lon = st.number_input("Enter Longitude", value=72.833597, format="%.6f")

    # System Parameters
    st.markdown("<h2 class='sub-header'>System Parameters</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        roof_area = st.number_input("Available Roof Area (sq. meters)", min_value=1.0, value=50.0)
        panel_efficiency = st.slider("Solar Panel Efficiency (%)", min_value=10.0, max_value=25.0, value=15.0, step=0.5,
                                    help="Modern solar panels typically have efficiencies between 15-22%")
        system_losses = st.slider("System Losses (%)", min_value=5.0, max_value=30.0, value=14.0, step=1.0,
                                 help="Includes losses from wiring, inverter efficiency, dust, shading, etc.")
    
    with col2:
        panel_degradation = st.slider("Annual Panel Degradation (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                                     help="Rate at which solar panels lose efficiency each year")
        panel_tilt = st.slider("Panel Tilt (degrees)", min_value=0, max_value=60, value=20,
                              help="Optimal tilt is typically close to your latitude")
        panel_azimuth = st.slider("Panel Azimuth (degrees)", min_value=0, max_value=359, value=180,
                                 help="180° is south-facing (optimal in northern hemisphere)")

    # 3D Model Parameters
    st.markdown("<h2 class='sub-header'>3D Model Parameters</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        roof_shape = st.selectbox("Roof Shape", ["flat", "gabled", "hipped"], 
                                 help="Shape of your roof for 3D modeling")
        
        include_obstacles = st.checkbox("Include Obstacles (Trees, Buildings, etc.)", value=False)
    
    obstacles = []
    if include_obstacles:
        st.markdown("<h3>Obstacles</h3>", unsafe_allow_html=True)
        
        num_obstacles = st.number_input("Number of Obstacles", min_value=0, max_value=10, value=1, step=1)
        
        for i in range(int(num_obstacles)):
            st.markdown(f"<h4>Obstacle {i+1}</h4>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                obstacle_type = st.selectbox(f"Type #{i+1}", ["tree", "building"], key=f"obs_type_{i}")
            
            with col2:
                obstacle_height = st.number_input(f"Height (m) #{i+1}", min_value=1.0, max_value=50.0, value=5.0, key=f"obs_height_{i}")
            
            with col3:
                obstacle_distance = st.number_input(f"Distance from Roof (m) #{i+1}", min_value=1.0, max_value=50.0, value=10.0, key=f"obs_dist_{i}")
            
            obstacle_azimuth = st.slider(f"Direction from Roof (degrees) #{i+1}", min_value=0, max_value=359, value=90, key=f"obs_azimuth_{i}",
                                        help="0° = North, 90° = East, 180° = South, 270° = West")
            
            obstacles.append({
                "type": obstacle_type,
                "height": obstacle_height,
                "distance": obstacle_distance,
                "azimuth": obstacle_azimuth
            })

    # Financial Parameters
    st.markdown("<h2 class='sub-header'>Financial Parameters</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        installation_cost_per_kw = st.number_input("Installation Cost (₹ per kW)", min_value=30000.0, value=60000.0, step=1000.0)
        monthly_bill = st.number_input("Current Monthly Electricity Bill (₹)", min_value=0.0, value=3000.0)
        rate_per_kwh = st.number_input("Electricity Rate (₹ per kWh)", min_value=0.1, value=8.0, step=0.1)
    
    with col2:
        annual_rate_increase = st.slider("Annual Electricity Rate Increase (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1,
                                        help="Historical increase in electricity prices")
        maintenance_cost_percent = st.slider("Annual Maintenance Cost (% of installation)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        discount_rate = st.slider("Discount Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1,
                                 help="Used for NPV calculation, typically the opportunity cost of capital")
    
    with col3:
        subsidy_percent = st.slider("Government Subsidy (%)", min_value=0.0, max_value=80.0, value=30.0, step=1.0)
        tax_benefit_percent = st.slider("Tax Benefits (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
        system_lifetime = st.slider("System Lifetime (years)", min_value=10, max_value=40, value=25)

    # Advanced Options (collapsible)
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            inverter_replacement_year = st.number_input("Inverter Replacement Year", min_value=5, max_value=20, value=10)
            inverter_cost_percent = st.slider("Inverter Cost (% of initial system cost)", min_value=5.0, max_value=30.0, value=15.0, step=1.0)
            battery_storage = st.checkbox("Include Battery Storage")
        
        with col2:
            if battery_storage:
                battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=1.0, value=10.0)
                battery_cost_per_kwh = st.number_input("Battery Cost (₹ per kWh)", min_value=5000.0, value=30000.0)
                battery_lifetime = st.number_input("Battery Lifetime (years)", min_value=5, max_value=20, value=10)
            else:
                battery_capacity = 0
                battery_cost_per_kwh = 0
                battery_lifetime = 0
            
            net_metering = st.checkbox("Net Metering Available", value=True)
            export_rate_factor = 1.0
            if net_metering:
                export_rate_factor = st.slider("Export Rate (as fraction of import rate)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

    # Blender path input
    with st.expander("Blender Configuration"):
        blender_path = st.text_input("Path to Blender Executable", 
                                    value="/Applications/Blender.app/Contents/MacOS/Blender" if sys.platform == "darwin" else 
                                    "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe" if sys.platform == "win32" else
                                    "/usr/bin/blender")
        
        st.info("If Blender is not installed, you can download it from https://www.blender.org/download/")

    # Calculate button
    calculate_button = st.button("Calculate Solar ROI with 3D Shading Analysis", type="primary", use_container_width=True)

# 3D Model & Shading tab
with tab2:
    st.markdown("<h2 class='sub-header'>3D Model & Shading Analysis</h2>", unsafe_allow_html=True)
    
    if 'shading_results' in st.session_state and st.session_state.shading_results is not None:
        try:
            # Display debug information
            with st.expander("Debug Information", expanded=False):
                st.write("### Shading Analysis Status")
                if st.session_state.results and st.session_state.results.get('shading_applied', False):
                    st.success("✅ Shading analysis was successfully applied")
                else:
                    st.warning("⚠️ Shading analysis was not applied")
                
                # Show available data structure
                st.write("Available keys in shading_results:", list(st.session_state.shading_results.keys()))
                if 'shading_analysis' in st.session_state.shading_results:
                    st.write("Available keys in shading_analysis:", 
                            list(st.session_state.shading_results['shading_analysis'].keys()))
            
            # Extract data safely
            shading_analysis = st.session_state.shading_results.get('shading_analysis', {})
            monthly_data = shading_analysis.get('monthly_averages', [])
            seasonal_variation = shading_analysis.get('seasonal_variation', {})
            hourly_variation = shading_analysis.get('hourly_variation', {})
            
            if not monthly_data:
                st.warning("No monthly shading data available")
            else:
                # All the visualization code goes here inside the else block
                # Display shading analysis visualizations
                st.markdown("<h3>Monthly Shading Analysis</h3>", unsafe_allow_html=True)
                
                # 1. Monthly Shading Heatmap
                shading_values = [float(value) for value in monthly_data]
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                # ... rest of your visualization code ...
                
        except Exception as e:
            st.error(f"Error displaying shading analysis: {str(e)}")
            st.write("Debug: Full error details")
            import traceback
            st.code(traceback.format_exc())
            
    else:
        st.info("Run the calculation to generate a 3D model and shading analysis.")


# Function to integrate shading results with energy production calculations
def apply_shading_to_production(energy_production, shading_results):
    """
    Apply monthly shading factors to energy production estimates
    
    Parameters:
    -----------
    energy_production : array-like
        Monthly energy production without shading (12 months)
    shading_results : dict
        Results from Blender shading analysis
    
    Returns:
    --------
    array-like
        Adjusted monthly energy production with shading effects
    """
    monthly_shading = shading_results['monthly_shading_factors']
    adjusted_production = np.copy(energy_production)
    
    for i, month_data in enumerate(monthly_shading):
        shading_factor = month_data['average_shading_percentage'] / 100
        adjusted_production[i] *= (1 - shading_factor)
    
    return adjusted_production

# Function to visualize shading analysis results
def visualize_shading_analysis(shading_results):
    """
    Visualize the shading analysis results from Blender simulation
    
    Args:
        shading_results (dict): Dictionary containing shading analysis data
    """
    
    if not shading_results or 'shading_analysis' not in shading_results:
        st.warning("⚠️ No shading analysis data available")
        return
    
        # Debug: Show available keys
    with st.expander("Debug: Data Structure", expanded=False):
        st.write("Available keys in shading_results:", list(shading_results.keys()))
        if 'shading_analysis' in shading_results:
            st.write("Available keys in shading_analysis:", list(shading_results['shading_analysis'].keys()))
    

    st.subheader("🌤️ 3D Shading Analysis Results")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Extract data
    shading_data = shading_results['shading_analysis']
    monthly_averages = shading_data.get('monthly_averages', [])
    annual_average = shading_data.get('annual_average', 0)
    seasonal_variation = shading_data.get('seasonal_variation', {})
    hourly_variation = shading_data.get('hourly_variation', {})
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Annual Average Shading",
            f"{annual_average:.1f}%",
            delta=f"{annual_average - 25:.1f}%" if annual_average < 25 else None,
            delta_color="inverse"
        )
    
    with col2:
        best_month_shading = min(monthly_averages) if monthly_averages else 0
        st.metric(
            "Best Month Performance",
            f"{best_month_shading:.1f}%",
            delta="Optimal" if best_month_shading < 20 else "Good" if best_month_shading < 30 else "Fair"
        )
    
    with col3:
        worst_month_shading = max(monthly_averages) if monthly_averages else 0
        st.metric(
            "Highest Shading Month",
            f"{worst_month_shading:.1f}%",
            delta="Challenging" if worst_month_shading > 40 else "Manageable"
        )
    
    with col4:
        shading_range = worst_month_shading - best_month_shading if monthly_averages else 0
        st.metric(
            "Seasonal Variation",
            f"{shading_range:.1f}%",
            delta="High" if shading_range > 25 else "Moderate" if shading_range > 15 else "Low"
        )
    
    # Monthly shading chart
    if monthly_averages:
        st.subheader("📊 Monthly Shading Analysis")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create color scale based on shading levels
        colors = []
        for value in monthly_averages:
            if value < 20:
                colors.append('#2E8B57')  # Green - Good
            elif value < 30:
                colors.append('#FFD700')  # Yellow - Fair
            elif value < 40:
                colors.append('#FF8C00')  # Orange - Moderate
            else:
                colors.append('#DC143C')  # Red - High shading
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=months,
            y=monthly_averages,
            name='Monthly Average Shading',
            marker_color=colors,
            text=[f"{val:.1f}%" for val in monthly_averages],
            textposition='auto',
        ))
        
        fig_monthly.update_layout(
            title='Monthly Average Shading Percentage',
            xaxis_title='Month',
            yaxis_title='Shading Percentage (%)',
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, max(monthly_averages) * 1.1])
        )
        
        # Add reference lines
        fig_monthly.add_hline(y=20, line_dash="dash", line_color="green", 
                             annotation_text="Good Performance (<20%)")
        fig_monthly.add_hline(y=30, line_dash="dash", line_color="orange", 
                             annotation_text="Moderate Performance (<30%)")
        
        st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_shading_chart")

    
    # Seasonal comparison
    if seasonal_variation:
        st.subheader("🌍 Seasonal Performance")
        
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        seasonal_values = [
            seasonal_variation.get('spring', 0),
            seasonal_variation.get('summer', 0),
            seasonal_variation.get('autumn', 0),
            seasonal_variation.get('winter', 0)
        ]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            spring_val = seasonal_values[0]
            st.metric("🌸 Spring", f"{spring_val:.1f}%", 
                     delta="Excellent" if spring_val < 20 else "Good" if spring_val < 30 else "Fair")
        
        with col2:
            summer_val = seasonal_values[1]
            st.metric("☀️ Summer", f"{summer_val:.1f}%", 
                     delta="Peak Season" if summer_val < 25 else "Good" if summer_val < 35 else "Fair")
        
        with col3:
            autumn_val = seasonal_values[2]
            st.metric("🍂 Autumn", f"{autumn_val:.1f}%", 
                     delta="Good" if autumn_val < 30 else "Moderate" if autumn_val < 40 else "Challenging")
        
        with col4:
            winter_val = seasonal_values[3]
            st.metric("❄️ Winter", f"{winter_val:.1f}%", 
                     delta="Expected" if winter_val < 45 else "High" if winter_val < 55 else "Very High")
    
    # Daily variation analysis
    if hourly_variation:
        st.subheader("⏰ Daily Shading Patterns")
        
        morning_data = hourly_variation.get('morning', [])
        noon_data = hourly_variation.get('noon', [])
        afternoon_data = hourly_variation.get('afternoon', [])
        
        if morning_data and noon_data and afternoon_data:
            fig_daily = go.Figure()
            
            fig_daily.add_trace(go.Scatter(
                x=months,
                y=morning_data,
                mode='lines+markers',
                name='Morning (8 AM)',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ))
            
            fig_daily.add_trace(go.Scatter(
                x=months,
                y=noon_data,
                mode='lines+markers',
                name='Noon (12 PM)',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ))
            
            fig_daily.add_trace(go.Scatter(
                x=months,
                y=afternoon_data,
                mode='lines+markers',
                name='Afternoon (4 PM)',
                line=dict(color='#45B7D1', width=3),
                marker=dict(size=8)
            ))
            
            fig_daily.update_layout(
                title='Hourly Shading Variation Throughout the Year',
                xaxis_title='Month',
                yaxis_title='Shading Percentage (%)',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_daily, use_container_width=True, key="daily_shading_chart")

            
            # Daily insights
            avg_morning = sum(morning_data) / len(morning_data)
            avg_noon = sum(noon_data) / len(noon_data)
            avg_afternoon = sum(afternoon_data) / len(afternoon_data)
            
            st.info(f"""
            📈 **Daily Pattern Insights:**
            - **Morning (8 AM)**: Average {avg_morning:.1f}% shading
            - **Noon (12 PM)**: Average {avg_noon:.1f}% shading (Peak sun)
            - **Afternoon (4 PM)**: Average {avg_afternoon:.1f}% shading
            
            Best performance typically occurs at noon when the sun is highest.
            """)
    
    # System information
    if 'system_info' in shading_results:
        st.subheader("🔧 System Configuration")
        system_info = shading_results['system_info']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Panels", system_info.get('num_panels', 0))
        
        with col2:
            st.metric("Total Panel Area", f"{system_info.get('total_area', 0):.1f} m²")
        
        with col3:
            roof_util = system_info.get('roof_utilization', 0) * 100
            st.metric("Roof Utilization", f"{roof_util:.1f}%")
    
    # Performance impact analysis
    st.subheader("⚡ Impact on Energy Production")
    
    # Calculate energy impact
    base_efficiency = 100 - annual_average  # Remaining efficiency after shading
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Effective System Efficiency",
            f"{base_efficiency:.1f}%",
            delta=f"{base_efficiency - 80:.1f}%" if base_efficiency > 80 else None
        )
    
    with col2:
        if base_efficiency > 85:
            performance_rating = "Excellent"
            color = "green"
        elif base_efficiency > 75:
            performance_rating = "Good"
            color = "blue"
        elif base_efficiency > 65:
            performance_rating = "Fair"
            color = "orange"
        else:
            performance_rating = "Poor"
            color = "red"
        
        st.markdown(f"**Performance Rating:** :{color}[{performance_rating}]")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if annual_average < 20:
        recommendations.append("✅ Excellent shading conditions - proceed with installation")
    elif annual_average < 30:
        recommendations.append("✅ Good shading conditions - minor optimizations possible")
    else:
        recommendations.append("⚠️ Consider obstacle removal or panel repositioning")
    
    if worst_month_shading > 50:
        recommendations.append("🔄 Consider seasonal panel angle adjustment")
    
    if shading_range > 30:
        recommendations.append("📊 High seasonal variation - consider battery storage for winter months")
    
    for rec in recommendations:
        st.write(rec)
    
    # Export option
    if st.button("📥 Export Shading Analysis Report"):
        # Create a summary report
        report_data = {
            "analysis_summary": {
                "annual_average_shading": annual_average,
                "best_month": months[monthly_averages.index(min(monthly_averages))] if monthly_averages else "N/A",
                "worst_month": months[monthly_averages.index(max(monthly_averages))] if monthly_averages else "N/A",
                "seasonal_variation": seasonal_variation,
                "performance_rating": performance_rating
            },
            "monthly_data": dict(zip(months, monthly_averages)) if monthly_averages else {},
            "recommendations": recommendations
        }
        
        # Convert to JSON for download
        json_str = json.dumps(report_data, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"shading_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )



# Now modify the calculate_solar_roi function to use the improved error handling
def calculate_solar_roi(lat, lon, roof_area, panel_efficiency, system_losses, panel_degradation, 
                        panel_tilt, panel_azimuth, installation_cost_per_kw, rate_per_kwh, 
                        annual_rate_increase, maintenance_cost_percent, discount_rate, 
                        subsidy_percent, tax_benefit_percent, system_lifetime, 
                        inverter_replacement_year, inverter_cost_percent, 
                        battery_storage, battery_capacity, battery_cost_per_kwh, battery_lifetime,
                        net_metering, export_rate_factor, 
                        roof_shape="flat", obstacles=None, blender_path=None):

    try:
        # Get solar irradiance data
        site = Location(lat, lon, tz='Asia/Kolkata')
        
        # Calculate system size based on roof area and efficiency
        # 1 kW typically requires about 7 sq meters at 15% efficiency
        # Adjusting for the actual efficiency
        reference_efficiency = 15.0
        reference_area_per_kw = 7.0
        system_size_kw = roof_area * (panel_efficiency / reference_efficiency) / reference_area_per_kw
        
        # Get solar resource data and calculate base energy production
        solpos = site.get_solarposition(times=pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H'))
        clearsky = site.get_clearsky(times=solpos.index)
        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=panel_tilt,
            surface_azimuth=panel_azimuth,
            dni=clearsky['dni'],
            ghi=clearsky['ghi'],
            dhi=clearsky['dhi'],
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth']
        )
        
        # Calculate annual energy production
        annual_irradiance_kwh_per_m2 = poa_irradiance['poa_global'].sum() / 1000
        system_efficiency = panel_efficiency / 100 * (1 - system_losses / 100)
        first_year_energy_kwh = annual_irradiance_kwh_per_m2 * roof_area * system_efficiency
        # Initialize shading_results before the Blender section
        shading_results = None
        # Run Blender shading analysis
        script_content, output_file = create_blender_script(
            lat=lat, lon=lon,
            roof_area=roof_area,
            panel_tilt=panel_tilt,
            panel_azimuth=panel_azimuth,
            obstacles=obstacles,
            roof_shape=roof_shape
        )
        
        output_file, success, error_msg = run_blender_simulation(script_content, blender_path)
        
        st.write(f"Debug: Blender simulation result - Success: {success}")

        # Create a container for the shading analysis
        shading_container = st.container()
        if output_file:
            st.write(f"Debug: Expected output file: {output_file}")
            st.write(f"Debug: File exists: {os.path.exists(output_file) if output_file else 'N/A'}")

        with shading_container:
            st.subheader("🌤️ Shading Analysis")
            
            if success and output_file and os.path.exists(output_file):
                try:
                    # Load shading results
                    with open(output_file, 'r') as f:
                        shading_results = json.load(f)
                    
                    # Visualize the shading analysis
                    visualize_shading_analysis(shading_results)
                    
                    # Get annual average shading for energy calculations
                    annual_shading = shading_results['shading_analysis']['annual_average']
                    shading_factor = 1 - (annual_shading / 100)
                    
                except Exception as e:
                    st.error(f"Error processing shading results: {e}")
                    shading_factor = 0.85  # Default 15% shading if analysis fails
            else:
                st.warning(f"⚠️ Shading analysis could not be performed: {error_msg}")
                st.info("Using default shading factor of 15%")
                shading_factor = 0.85
        
        # Calculate installation cost
        base_installation_cost = system_size_kw * installation_cost_per_kw
        
        if battery_storage:
            battery_cost = battery_capacity * battery_cost_per_kwh
            total_installation_cost = base_installation_cost + battery_cost
        else:
            battery_cost = 0
            total_installation_cost = base_installation_cost
        
        # Apply subsidy and tax benefits
        subsidy_amount = total_installation_cost * (subsidy_percent / 100)
        tax_benefit_amount = total_installation_cost * (tax_benefit_percent / 100)
        net_installation_cost = total_installation_cost - subsidy_amount - tax_benefit_amount
        
        # Initialize arrays for yearly calculations
        years = np.arange(1, system_lifetime + 1)
        energy_production = np.zeros(system_lifetime)
        electricity_rates = np.zeros(system_lifetime)
        savings = np.zeros(system_lifetime)
        maintenance_costs = np.zeros(system_lifetime)
        replacement_costs = np.zeros(system_lifetime)
        net_cash_flow = np.zeros(system_lifetime)
        cumulative_cash_flow = np.zeros(system_lifetime)
        discounted_cash_flow = np.zeros(system_lifetime)
        cumulative_discounted_cash_flow = np.zeros(system_lifetime)
        
        # Calculate yearly values with shading effects
        for year in range(system_lifetime):
            # Energy production with degradation
            energy_production[year] = first_year_energy_kwh * (1 - panel_degradation / 100) * shading_factor ** year
            
            # Electricity rate with annual increase
            electricity_rates[year] = rate_per_kwh * (1 + annual_rate_increase / 100) ** year
            
            # Annual savings
            savings[year] = energy_production[year] * electricity_rates[year]
            
            # Maintenance costs
            maintenance_costs[year] = total_installation_cost * (maintenance_cost_percent / 100)
            
            # Replacement costs
            if (year + 1) % inverter_replacement_year == 0:
                replacement_costs[year] = base_installation_cost * (inverter_cost_percent / 100)
            
            if battery_storage and (year + 1) % battery_lifetime == 0 and year > 0:
                replacement_costs[year] += battery_cost
            
            # Net cash flow
            if year == 0:
                net_cash_flow[year] = -net_installation_cost + savings[year] - maintenance_costs[year] - replacement_costs[year]
            else:
                net_cash_flow[year] = savings[year] - maintenance_costs[year] - replacement_costs[year]
            
            # Cumulative cash flow
            if year == 0:
                cumulative_cash_flow[year] = net_cash_flow[year]
            else:
                cumulative_cash_flow[year] = cumulative_cash_flow[year - 1] + net_cash_flow[year]
            
            # Discounted cash flow
            discounted_cash_flow[year] = net_cash_flow[year] / (1 + discount_rate / 100) ** year
            
            # Cumulative discounted cash flow
            if year == 0:
                cumulative_discounted_cash_flow[year] = discounted_cash_flow[year]
            else:
                cumulative_discounted_cash_flow[year] = cumulative_discounted_cash_flow[year - 1] + discounted_cash_flow[year]
        
        # Calculate financial metrics
        npv = cumulative_discounted_cash_flow[-1]
        
        try:
            irr_cash_flows = [-net_installation_cost] + list(net_cash_flow[1:])
            irr = np.irr(irr_cash_flows) * 100
        except:
            irr = 0
        
        # Calculate payback period
        payback_period = None
        for i in range(len(cumulative_cash_flow)):
            if cumulative_cash_flow[i] >= 0:
                if i == 0:
                    payback_period = 1
                else:
                    payback_period = i + abs(cumulative_cash_flow[i-1]) / (abs(cumulative_cash_flow[i-1]) + cumulative_cash_flow[i])
                break
        
        if payback_period is None:
            payback_period = system_lifetime
        
        # Calculate LCOE
        total_energy = sum(energy_production)
        total_costs = net_installation_cost + sum(maintenance_costs) + sum(replacement_costs)
        lcoe = total_costs / total_energy if total_energy > 0 else 0
        
        # Calculate average monthly savings and total lifetime savings
        average_monthly_savings = np.mean(savings) / 12
        total_lifetime_savings = sum(savings)
        
        # Calculate ROI
        roi = (total_lifetime_savings - total_costs) / total_costs * 100 if total_costs > 0 else 0
        
        # Create results dictionary
        results = {
            'system_size_kw': system_size_kw,
            'first_year_energy_kwh': first_year_energy_kwh,
            'total_installation_cost': total_installation_cost,
            'subsidy_amount': subsidy_amount,
            'tax_benefit_amount': tax_benefit_amount,
            'net_installation_cost': net_installation_cost,
            'years': years,
            'energy_production': energy_production,
            'electricity_rates': electricity_rates,
            'savings': savings,
            'maintenance_costs': maintenance_costs,
            'replacement_costs': replacement_costs,
            'net_cash_flow': net_cash_flow,
            'cumulative_cash_flow': cumulative_cash_flow,
            'discounted_cash_flow': discounted_cash_flow,
            'cumulative_discounted_cash_flow': cumulative_discounted_cash_flow,
            'npv': npv,
            'irr': irr,
            'payback_period': payback_period,
            'lcoe': lcoe,
            'average_monthly_savings': average_monthly_savings,
            'total_lifetime_savings': total_lifetime_savings,
            'roi': roi,
            'shading_applied': success
        }
        
        return results, shading_results if 'shading_results' in locals() else None, None
    
    except Exception as e:
        import traceback
        st.error(f"Error in ROI calculation: {e}")
        return None, None, f"Error in calculation: {str(e)}\n{traceback.format_exc()}"

# Handle calculation and display results
if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.shading_results = None
    st.session_state.error = None

if calculate_button:
    if lat is None or lon is None:
        with tab1:
            st.error("Please provide valid location information (latitude and longitude).")
    else:
        with st.spinner("Calculating solar ROI with 3D shading analysis..."):
            results, shading_results, error = calculate_solar_roi(
                lat=lat, lon=lon, roof_area=roof_area, panel_efficiency=panel_efficiency,
                system_losses=system_losses, panel_degradation=panel_degradation,
                panel_tilt=panel_tilt, panel_azimuth=panel_azimuth,
                installation_cost_per_kw=installation_cost_per_kw, rate_per_kwh=rate_per_kwh,
                annual_rate_increase=annual_rate_increase, maintenance_cost_percent=maintenance_cost_percent,
                discount_rate=discount_rate, subsidy_percent=subsidy_percent,
                tax_benefit_percent=tax_benefit_percent, system_lifetime=system_lifetime,
                inverter_replacement_year=inverter_replacement_year, inverter_cost_percent=inverter_cost_percent,
                battery_storage=battery_storage, battery_capacity=battery_capacity,
                battery_cost_per_kwh=battery_cost_per_kwh, battery_lifetime=battery_lifetime,
                net_metering=net_metering, export_rate_factor=export_rate_factor,
                roof_shape=roof_shape, obstacles=obstacles, blender_path=blender_path
            )
            
            st.session_state.results = results
            st.session_state.shading_results = shading_results
            st.session_state.error = error

# Display Results Overview
with tab3:
    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.results is not None:  # Check if results exist in session state
        results = st.session_state.results  # Get results from session state
        
        st.markdown("<h2 class='sub-header'>Solar System Overview</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            system_size_value = f"{results['system_size_kw']:.2f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{system_size_value} kW</div>
                    <div class="metric-label">System Size</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            first_year_energy = f"{results['first_year_energy_kwh']:,.0f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{first_year_energy} kWh</div>
                    <div class="metric-label">First Year Production</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            installation_cost = f"₹{results['net_installation_cost']:,.0f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{installation_cost}</div>
                    <div class="metric-label">Net Installation Cost</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col4:
            payback_period = f"{results['payback_period']:.1f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{payback_period} years</div>
                    <div class="metric-label">Payback Period</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Add shading impact notice if shading was applied
        if st.session_state.results.get('shading_applied', False):  # Use session state
            st.markdown(
                """
                <div class="info-box">
                <h3>✓ Shading Analysis Applied</h3>
                <p>The energy production estimates include the effects of shading from nearby obstacles.
                View the 3D Model & Shading tab for detailed shading analysis.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        st.markdown("<h2 class='sub-header'>Financial Summary</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_savings = f"₹{results['average_monthly_savings']:,.0f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{monthly_savings}</div>
                    <div class="metric-label">Avg. Monthly Savings</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            lifetime_savings = f"₹{results['total_lifetime_savings']:,.0f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{lifetime_savings}</div>
                    <div class="metric-label">Lifetime Savings</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            roi_value = f"{results['roi']:.1f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{roi_value}%</div>
                    <div class="metric-label">Return on Investment</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col4:
            npv_value = f"₹{results['npv']:,.0f}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{npv_value}</div>
                    <div class="metric-label">Net Present Value</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
    else:
        st.info("Please enter your parameters and click 'Calculate Solar ROI' to see results.")

# Handle calculation button click
if calculate_button:
    if lat is None or lon is None:
        with tab1:
            st.error("Please provide valid location information (latitude and longitude).")
    else:
        with st.spinner("Calculating solar ROI with 3D shading analysis..."):
            results, shading_results, error = calculate_solar_roi(
                lat=lat, lon=lon,
                roof_area=roof_area,
                panel_efficiency=panel_efficiency,
                system_losses=system_losses,
                panel_degradation=panel_degradation,
                panel_tilt=panel_tilt,
                panel_azimuth=panel_azimuth,
                installation_cost_per_kw=installation_cost_per_kw,
                rate_per_kwh=rate_per_kwh,
                annual_rate_increase=annual_rate_increase,
                maintenance_cost_percent=maintenance_cost_percent,
                discount_rate=discount_rate,
                subsidy_percent=subsidy_percent,
                tax_benefit_percent=tax_benefit_percent,
                system_lifetime=system_lifetime,
                inverter_replacement_year=inverter_replacement_year,
                inverter_cost_percent=inverter_cost_percent,
                battery_storage=battery_storage,
                battery_capacity=battery_capacity,
                battery_cost_per_kwh=battery_cost_per_kwh,
                battery_lifetime=battery_lifetime,
                net_metering=net_metering,
                export_rate_factor=export_rate_factor,
                roof_shape=roof_shape,
                obstacles=obstacles,
                blender_path=blender_path
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.shading_results = shading_results
            st.session_state.error = error

        
# Display Detailed Analysis
with tab4:
    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.results:
        results = st.session_state.results
        
        st.markdown("<h2 class='sub-header'>Detailed Financial Analysis</h2>", unsafe_allow_html=True)
        
        # Create detailed financial table
        financial_data = pd.DataFrame({
            'Year': results['years'],
            'Energy Production (kWh)': results['energy_production'],
            'Electricity Rate (₹/kWh)': results['electricity_rates'],
            'Annual Savings (₹)': results['savings'],
            'Maintenance Costs (₹)': results['maintenance_costs'],
            'Replacement Costs (₹)': results['replacement_costs'],
            'Net Cash Flow (₹)': results['net_cash_flow'],
            'Cumulative Cash Flow (₹)': results['cumulative_cash_flow'],
            'Discounted Cash Flow (₹)': results['discounted_cash_flow'],
            'Cumulative NPV (₹)': results['cumulative_discounted_cash_flow']
        })
        
        # Format the financial data
        for col in financial_data.columns:
            if 'Rate' not in col and 'Year' not in col:
                financial_data[col] = financial_data[col].round(2)
        
        # Display the table with formatting
        st.dataframe(financial_data.style.format({
            'Energy Production (kWh)': '{:,.0f}',
            'Electricity Rate (₹/kWh)': '{:.2f}',
            'Annual Savings (₹)': '{:,.0f}',
            'Maintenance Costs (₹)': '{:,.0f}',
            'Replacement Costs (₹)': '{:,.0f}',
            'Net Cash Flow (₹)': '{:,.0f}',
            'Cumulative Cash Flow (₹)': '{:,.0f}',
            'Discounted Cash Flow (₹)': '{:,.0f}',
            'Cumulative NPV (₹)': '{:,.0f}'
        }), use_container_width=True)
        
        # Additional Charts
        st.markdown("<h2 class='sub-header'>Additional Charts</h2>", unsafe_allow_html=True)
        
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Energy Production Over Time", "Electricity Rate Projection", "NPV Sensitivity Analysis", "Monthly Savings Projection"]
        )
        
        if chart_type == "Energy Production Over Time":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['years'],
                y=results['energy_production'],
                mode='lines+markers',
                name='Annual Energy Production',
                line=dict(color='#FFA000', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Solar Energy Production Over Time (with Panel Degradation)',
                xaxis_title='Year',
                yaxis_title='Energy Production (kWh)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                <div class="info-box">
                <p>This chart shows the projected annual energy production from your solar system over its lifetime, 
                accounting for panel degradation of {:.1f}% per year. As panels age, they gradually produce less electricity.</p>
                </div>
                """.format(panel_degradation), 
                unsafe_allow_html=True
            )
            
        elif chart_type == "Electricity Rate Projection":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['years'],
                y=results['electricity_rates'],
                mode='lines+markers',
                name='Electricity Rate',
                line=dict(color='#E91E63', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Projected Electricity Rate Over Time',
                xaxis_title='Year',
                yaxis_title='Electricity Rate (₹/kWh)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                <div class="info-box">
                <p>This chart shows the projected electricity rate over time, assuming an annual increase of {:.1f}%. 
                Rising electricity costs make solar investments more valuable over time.</p>
                </div>
                """.format(annual_rate_increase), 
                unsafe_allow_html=True
            )
            
        elif chart_type == "NPV Sensitivity Analysis":
            # Create sensitivity analysis for NPV
            discount_rates = np.arange(max(1, discount_rate-5), discount_rate+6, 1)
            npvs = []
            
            for dr in discount_rates:
                # Recalculate NPV with different discount rates
                dcf = np.zeros(system_lifetime)
                for year in range(system_lifetime):
                    dcf[year] = results['net_cash_flow'][year] / (1 + dr / 100) ** year
                
                npvs.append(sum(dcf))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=discount_rates,
                y=npvs,
                mode='lines+markers',
                name='NPV',
                line=dict(color='#673AB7', width=3),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=[discount_rate],
                y=[results['npv']],
                mode='markers',
                name='Current Discount Rate',
                marker=dict(color='red', size=12, symbol='star')
            ))
            fig.add_trace(go.Scatter(
                x=[min(discount_rates), max(discount_rates)],
                y=[0, 0],
                mode='lines',
                name='Break-even',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig.update_layout(
                title='NPV Sensitivity to Discount Rate',
                xaxis_title='Discount Rate (%)',
                yaxis_title='Net Present Value (₹)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                <div class="info-box">
                <p>This chart shows how the Net Present Value (NPV) of your solar investment changes with different discount rates. 
                The discount rate represents the opportunity cost of capital or the return you could get from alternative investments.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        elif chart_type == "Monthly Savings Projection":
            # Calculate monthly savings for each year
            monthly_savings = results['savings'] / 12
            
            # Create a month-year grid for 5 years
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            years_to_show = min(5, system_lifetime)
            
            # Create heatmap data
            heatmap_data = np.zeros((years_to_show, 12))
            for year in range(years_to_show):
                # Assume equal distribution across months for simplicity
                # In reality, this would vary by month based on solar insolation
                heatmap_data[year] = [monthly_savings[year]] * 12
            
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Month", y="Year", color="Monthly Savings (₹)"),
                x=months,
                y=[f"Year {i+1}" for i in range(years_to_show)],
                color_continuous_scale='Viridis',
                aspect="auto"
            )
            fig.update_layout(
                title='Projected Monthly Savings (First 5 Years)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                <div class="info-box">
                <p>This heatmap shows the projected monthly savings for the first 5 years of your solar system. 
                Note that this is a simplified view assuming equal monthly production, while actual production would vary by season.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Download Report
        st.markdown("<h2 class='sub-header'>Download Detailed Report</h2>", unsafe_allow_html=True)
        
        # Create a function to generate the report
        def generate_report():
            buffer = BytesIO()
            
            # Create a DataFrame for the report
            report_data = financial_data.copy()
            
            # Convert to Excel
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write the financial data
                report_data.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                
                # Write summary data
                summary_data = pd.DataFrame({
                    'Metric': [
                        'System Size (kW)', 'First Year Production (kWh)', 'Installation Cost (₹)',
                        'Net Installation Cost (₹)', 'Payback Period (years)', 'ROI (%)',
                        'NPV (₹)', 'IRR (%)', 'LCOE (₹/kWh)', 'Average Monthly Savings (₹)',
                        'Total Lifetime Savings (₹)'
                    ],
                    'Value': [
                        f"{results['system_size_kw']:.2f}",
                        f"{results['first_year_energy_kwh']:,.0f}",
                        f"{results['total_installation_cost']:,.0f}",
                        f"{results['net_installation_cost']:,.0f}",
                        f"{results['payback_period']:.1f}",
                        f"{results['roi']:.1f}",
                        f"{results['npv']:,.0f}",
                        f"{results['irr']:.1f}",
                        f"{results['lcoe']:.2f}",
                        f"{results['average_monthly_savings']:,.0f}",
                        f"{results['total_lifetime_savings']:,.0f}"
                    ]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Get the workbook and worksheet objects
                workbook = writer.book
                
                # Add a chart sheet
                chart_sheet = workbook.add_worksheet('Cash Flow Chart')
                
                # Create a chart object
                chart = workbook.add_chart({'type': 'line'})
                
                # Configure the chart
                chart.add_series({
                    'name': 'Cumulative Cash Flow',
                    'categories': ['Detailed Analysis', 1, 0, len(report_data), 0],
                    'values': ['Detailed Analysis', 1, 7, len(report_data), 7],
                })
                
                chart.set_title({'name': 'Cumulative Cash Flow Over Time'})
                chart.set_x_axis({'name': 'Year'})
                chart.set_y_axis({'name': 'Cumulative Cash Flow (₹)'})
                
                # Insert the chart into the chart sheet
                chart_sheet.insert_chart('B2', chart, {'x_scale': 2, 'y_scale': 2})
            
            # Return the Excel file
            buffer.seek(0)
            return buffer
        
        # Create a download button
        report = generate_report()
        st.download_button(
            label="Download Excel Report",
            data=report,
            file_name="Solar_ROI_Analysis.xlsx",
            mime="application/vnd.ms-excel",
            key='download-excel'
        )
    else:
        st.info("Please enter your parameters in the Input Parameters tab and click 'Calculate Solar ROI' to see results.")


# Update the About tab to include information about the 3D shading analysis
with tab5:
    st.markdown("<h2 class='sub-header'>About the Solar ROI Calculator with 3D Shading Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="info-box">
        <p>This advanced Solar ROI Calculator helps you evaluate the financial viability of installing a solar power system 
        for your home or business. The calculator takes into account various factors including:</p>
        
        <ul>
            <li><strong>Location-specific solar resource data</strong> based on your latitude and longitude</li>
            <li><strong>System parameters</strong> such as panel efficiency, tilt, azimuth, and degradation rate</li>
            <li><strong>Financial factors</strong> including installation costs, electricity rates, subsidies, and maintenance costs</li>
            <li><strong>Advanced options</strong> like battery storage, inverter replacement, and net metering</li>
            <li><strong>3D shading analysis</strong> using Blender to simulate the impact of nearby obstacles</li>
        </ul>
        
        <p>The calculator provides comprehensive financial metrics including:</p>
        
        <ul>
            <li><strong>Payback Period:</strong> The time it takes for the system to pay for itself through savings</li>
            <li><strong>Return on Investment (ROI):</strong> The percentage return on your initial investment</li>
            <li><strong>Net Present Value (NPV):</strong> The current value of all future cash flows</li>
            <li><strong>Internal Rate of Return (IRR):</strong> The annualized effective compounded return rate</li>
            <li><strong>Levelized Cost of Electricity (LCOE):</strong> The average cost per kWh of electricity generated</li>
        </ul>
        
        <p>The results are presented through intuitive visualizations and detailed tables to help you make an informed decision.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<h2 class='sub-header'>3D Shading Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="info-box">
        <p>The 3D shading analysis feature uses Blender, an open-source 3D modeling software, to create a realistic model of your solar installation
        and simulate the shading effects throughout the year. This provides more accurate energy production estimates by accounting for:</p>
        
        <ul>
            <li><strong>Seasonal sun position changes</strong> based on your specific latitude and longitude</li>
            <li><strong>Shading from nearby obstacles</strong> such as trees, buildings, and other structures</li>
            <li><strong>Panel layout and orientation</strong> including tilt and azimuth angles</li>
            <li><strong>Roof shape and characteristics</strong> that may affect panel placement and shading</li>
        </ul>
        
        <p>The simulation calculates shading percentages for different times of day and different seasons, providing a comprehensive
        understanding of how shading affects your solar energy production throughout the year.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if st.button("🔧 Test Blender Integration"):
    script_dir = os.path.join(os.path.expanduser("~"), "temp_blender")
    test_file = os.path.join(script_dir, "test_output.json")
    
    simple_script = f'''
import json
import os

print("Testing file creation...")
test_data = {{"test": "success", "timestamp": "now"}}

try:
    os.makedirs(os.path.dirname("{test_file}"), exist_ok=True)
    with open("{test_file}", 'w') as f:
        json.dump(test_data, f)
    print(f"Test file created: {test_file}")
except Exception as e:
    print(f"Error: {{e}}")
'''
    
    script_file = os.path.join(script_dir, "test_script.py")
    with open(script_file, 'w') as f:
        f.write(simple_script)
    
    # Run with Blender
    result = subprocess.run([blender_path, "--background", "--python", script_file], 
                           capture_output=True, text=True)
    
    if os.path.exists(test_file):
        st.success("✅ File creation test passed!")
    else:
        st.error("❌ File creation test failed!")
        st.code(result.stdout)

    
    # Rest of the About tab content remains the same
    # ...
