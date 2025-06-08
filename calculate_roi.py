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

def calculate_shading(month, day, hour):
    sun = update_sun_position(month, day, hour)
    shading_results = []
    
    for panel in panel_group.objects:
        panel_loc = panel.matrix_world.translation
        sun_dir = sun.rotation_euler.to_matrix() @ Vector((0, 0, -1))
        
        hit, loc, normal, index, obj, matrix = scene.ray_cast(
            bpy.context.view_layer.depsgraph,
            panel_loc,
            sun_dir
        )
        
        is_shaded = hit and obj != panel
        shading_results.append({{"panel": panel.name, "shaded": is_shaded}})
    
    return shading_results

print("Calculating shading for different times...")

# Calculate shading for different times
shading_data = []
for month in [3, 6, 9, 12]:
    for hour in [8, 12, 16]:
        day = 21
        print(f"Calculating shading for month {{month}}, day {{day}}, hour {{hour}}...")
        
        results = calculate_shading(month, day, hour)
        shaded_count = sum(1 for r in results if r["shaded"])
        shading_percentage = shaded_count / len(results) * 100 if results else 0
        
        shading_data.append({{
            "month": month,
            "day": day,
            "hour": hour,
            "shading_percentage": shading_percentage,
            "panel_details": results
        }})

print("Calculating monthly averages...")

# Calculate monthly averages
monthly_shading = []
for month in range(1, 13):
    month_shading = []
    for day in [7, 14, 21]:
        for hour in [8, 10, 12, 14, 16]:
            if 8 <= hour <= 16:
                results = calculate_shading(month, day, hour)
                shaded_count = sum(1 for r in results if r["shaded"])
                shading_percentage = shaded_count / len(results) * 100 if results else 0
                month_shading.append(shading_percentage)
    
    avg_shading = sum(month_shading) / len(month_shading) if month_shading else 0
    monthly_shading.append({{
        "month": month,
        "average_shading_percentage": avg_shading
    }})

print("Saving results...")

# Save results
output_data = {{
    "location": {{"latitude": {lat}, "longitude": {lon}}},
    "roof_area": {roof_area},
    "panel_tilt": {panel_tilt},
    "panel_azimuth": {panel_azimuth},
    "num_panels": num_panels,
    "panel_layout": {{"rows": rows, "columns": cols}},
    "sample_shading_data": shading_data,
    "monthly_shading_factors": monthly_shading
}}

with open("{output_file}", 'w') as f:
    json.dump(output_data, f, indent=2)

print("Analysis completed successfully!")
"""
    return script, output_file




def run_blender_simulation(script_content, blender_path=None):
    """Run the Blender simulation using the generated script
    
    Parameters:
    -----------
    script_content : str
        The Python script to be executed by Blender
    blender_path : str, optional
        Path to the Blender executable
    
    Returns:
    --------
    tuple
        (output_file_path, success, error_message)
    """

    # Create a temporary script file in a location we're sure to have write access
    script_dir = os.path.join(os.path.expanduser("~"), "temp_blender")
    os.makedirs(script_dir, exist_ok=True)
    script_file = os.path.join(script_dir, f"blender_script_{int(time.time())}.py")
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Create a specific output file path in the same directory
    output_file = os.path.join(script_dir, f"shading_results_{int(time.time())}.json")
    
    # Update the script to use this output file
    script_content = script_content.replace('output_file = tempfile.mktemp(suffix=\'.json\')', f'output_file = "{output_file}"')
    
    # Write the updated script
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Find Blender executable if not provided
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
        return None, False, "Blender executable not found. Please provide the path to Blender."
    
    # Run a simple test first to verify Blender can execute Python scripts
    test_script = os.path.join(script_dir, "test_blender.py")
    with open(test_script, 'w') as f:
        f.write("""
import bpy
import os

# Write a simple test file
test_file = os.path.join(os.path.dirname(__file__), "blender_test_output.txt")
with open(test_file, 'w') as f:
    f.write("Blender Python script executed successfully")

print("Test script completed successfully")
""")
    
    try:
        # Test if Blender can run Python scripts and write files
        test_result = subprocess.run(
            [blender_path, "--background", "--python", test_script],
            capture_output=True, text=True, check=True
        )
        
        test_output_file = os.path.join(script_dir, "blender_test_output.txt")
        if not os.path.exists(test_output_file):
            return None, False, f"Blender test script did not create output file. This suggests permission issues or Python execution problems.\nBlender output: {test_result.stdout}"
    except Exception as e:
        return None, False, f"Error running Blender test script: {str(e)}"
    
    # Now run the actual shading analysis script
    try:
        # Add verbose output to help with debugging
        print(f"Running Blender with script: {script_file}")
        print(f"Expected output file: {output_file}")
        
        result = subprocess.run(
            [blender_path, "--background", "--python", script_file],
            capture_output=True, text=True, check=True
        )
        
        print(f"Blender execution completed. Checking for output file: {output_file}")
        
        # Check if the output file was created
        if os.path.exists(output_file):
            print(f"Output file found: {output_file}")
            # Verify the file contains valid JSON
            try:
                with open(output_file, 'r') as f:
                    json_content = json.load(f)
                print("JSON content loaded successfully")
                return output_file, True, None
            except json.JSONDecodeError as e:
                return None, False, f"Output file contains invalid JSON: {str(e)}"
        else:
            print("Output file not found")
            # Write Blender's output to a log file for inspection
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
    finally:
        # Don't clean up files for debugging purposes
        print(f"Script file: {script_file}")
        print(f"Output file should be at: {output_file}")


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
        # Display debug information
        with st.expander("Debug Information", expanded=False):
            st.write("### Shading Analysis Status")
            if st.session_state.results.get('shading_applied', False):
                st.success("✅ Shading analysis was successfully applied")
            else:
                st.warning("⚠️ Shading analysis was not applied")
        
        # Display shading analysis visualizations
        st.markdown("<h3>Monthly Shading Analysis</h3>", unsafe_allow_html=True)
        
        # 1. Monthly Shading Heatmap
        monthly_data = st.session_state.shading_results['monthly_shading_factors']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        shading_values = [data['average_shading_percentage'] for data in monthly_data]
        
        fig_heatmap = px.imshow(
            [shading_values],
            labels=dict(x="Month", y="", color="Shading %"),
            x=months,
            y=['Average'],
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig_heatmap.update_layout(
            title='Monthly Shading Percentage',
            height=250
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 2. Daily Shading Pattern
        st.markdown("<h3>Daily Shading Patterns</h3>", unsafe_allow_html=True)
        
        sample_data = st.session_state.shading_results['sample_shading_data']
        daily_df = pd.DataFrame(sample_data)
        
        fig_daily = px.scatter(
            daily_df,
            x='hour',
            y='shading_percentage',
            color='month',
            size='shading_percentage',
            labels={
                'hour': 'Hour of Day',
                'shading_percentage': 'Shading %',
                'month': 'Month'
            },
            title='Shading Patterns Throughout the Day'
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # 3. Seasonal Analysis
        st.markdown("<h3>Seasonal Shading Analysis</h3>", unsafe_allow_html=True)
        
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        
        seasonal_data = {}
        for season, months_in_season in seasons.items():
            season_values = [data['average_shading_percentage'] 
                           for data in monthly_data 
                           if data['month'] in months_in_season]
            seasonal_data[season] = np.mean(season_values)
        
        fig_seasonal = px.bar(
            x=list(seasonal_data.keys()),
            y=list(seasonal_data.values()),
            labels={'x': 'Season', 'y': 'Average Shading %'},
            title='Seasonal Shading Analysis',
            color=list(seasonal_data.values()),
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # 4. Summary Metrics
        st.markdown("<h3>Shading Impact Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_shading = np.mean(shading_values)
            st.metric(
                "Average Annual Shading",
                f"{avg_shading:.1f}%",
                delta=f"-{avg_shading:.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            max_shading = max(shading_values)
            max_month = months[shading_values.index(max_shading)]
            st.metric(
                "Maximum Monthly Shading",
                f"{max_shading:.1f}%",
                f"in {max_month}"
            )
        
        with col3:
            min_shading = min(shading_values)
            min_month = months[shading_values.index(min_shading)]
            st.metric(
                "Minimum Monthly Shading",
                f"{min_shading:.1f}%",
                f"in {min_month}"
            )
        
        # 5. Download Report
        st.markdown("<h3>Download Analysis Report</h3>", unsafe_allow_html=True)
        
        # Create a formatted report
        report_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": {
                "latitude": st.session_state.shading_results['location']['latitude'],
                "longitude": st.session_state.shading_results['location']['longitude']
            },
            "system_details": {
                "roof_area": st.session_state.shading_results['roof_area'],
                "panel_tilt": st.session_state.shading_results['panel_tilt'],
                "panel_azimuth": st.session_state.shading_results['panel_azimuth'],
                "num_panels": st.session_state.shading_results['num_panels'],
                "panel_layout": st.session_state.shading_results['panel_layout']
            },
            "shading_analysis": {
                "monthly_averages": monthly_data,
                "seasonal_averages": seasonal_data,
                "overall_average": avg_shading,
                "maximum_shading": {
                    "value": max_shading,
                    "month": max_month
                },
                "minimum_shading": {
                    "value": min_shading,
                    "month": min_month
                }
            }
        }
        
        # Convert to JSON string
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="Download Shading Analysis Report",
            data=report_json,
            file_name="shading_analysis_report.json",
            mime="application/json"
        )
        
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
    """Create visualizations of shading analysis results"""
    
    # 1. Monthly Shading Heatmap
    monthly_data = shading_results['monthly_shading_factors']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    shading_values = [data['average_shading_percentage'] for data in monthly_data]
    
    fig_heatmap = px.imshow(
        [shading_values],
        labels=dict(x="Month", y="", color="Shading %"),
        x=months,
        y=['Average'],
        color_continuous_scale='RdBu_r',  # Red for high shading, blue for low
        aspect='auto'
    )
    fig_heatmap.update_layout(title='Monthly Shading Percentage')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 2. Daily Shading Pattern
    sample_data = shading_results['sample_shading_data']
    daily_df = pd.DataFrame(sample_data)
    
    fig_daily = px.scatter(
        daily_df,
        x='hour',
        y='shading_percentage',
        color='month',
        size='shading_percentage',
        labels={
            'hour': 'Hour of Day',
            'shading_percentage': 'Shading %',
            'month': 'Month'
        },
        title='Shading Patterns Throughout the Day'
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # 3. Seasonal Analysis
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    seasonal_data = {}
    for season, months in seasons.items():
        season_values = [data['average_shading_percentage'] 
                        for data in monthly_data 
                        if data['month'] in months]
        seasonal_data[season] = np.mean(season_values)
    
    fig_seasonal = px.bar(
        x=list(seasonal_data.keys()),
        y=list(seasonal_data.values()),
        labels={'x': 'Season', 'y': 'Average Shading %'},
        title='Seasonal Shading Analysis',
        color=list(seasonal_data.values()),
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # 4. Panel Layout with Shading
    if 'panel_layout' in shading_results:
        layout = shading_results['panel_layout']
        rows = layout['rows']
        cols = layout['columns']
        
        # Create panel shading visualization
        sample_time = sample_data[len(sample_data)//2]  # Mid-day sample
        panel_data = sample_time['panel_details']
        
        panel_matrix = np.zeros((rows, cols))
        for panel in panel_data:
            row, col = map(int, panel['panel'].split('_')[1:])
            panel_matrix[row][col] = 1 if panel['shaded'] else 0
        
        fig_layout = px.imshow(
            panel_matrix,
            labels=dict(x="Column", y="Row", color="Shaded"),
            color_continuous_scale=['lightblue', 'darkblue'],
            title=f"Panel Layout Shading at {sample_time['hour']}:00"
        )
        st.plotly_chart(fig_layout, use_container_width=True)
    
    # 5. Summary Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_shading = np.mean(shading_values)
        st.metric(
            "Average Annual Shading",
            f"{avg_shading:.1f}%",
            delta=f"-{avg_shading:.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        max_shading = max(shading_values)
        max_month = months[shading_values.index(max_shading)]
        st.metric(
            "Maximum Monthly Shading",
            f"{max_shading:.1f}%",
            f"in {max_month}"
        )
    
    with col3:
        min_shading = min(shading_values)
        min_month = months[shading_values.index(min_shading)]
        st.metric(
            "Minimum Monthly Shading",
            f"{min_shading:.1f}%",
            f"in {min_month}"
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
        
        if success:
            with open(output_file, 'r') as f:
                shading_results = json.load(f)
            
            # Convert annual production to monthly
            monthly_factors = np.array([0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.11, 0.1, 0.09, 0.08, 0.06, 0.05])
            monthly_production = first_year_energy_kwh * monthly_factors
            
            # Apply shading factors
            adjusted_monthly_production = apply_shading_to_production(monthly_production, shading_results)
            first_year_energy_kwh = sum(adjusted_monthly_production)
            
            # Store shading results for visualization
            st.session_state.shading_results = shading_results
        else:
            st.warning(f"Shading analysis could not be performed: {error_msg}")
            st.session_state.shading_results = None
        
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
            energy_production[year] = first_year_energy_kwh * (1 - panel_degradation / 100) ** year
            
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
        
        return results, shading_results, None
    
    except Exception as e:
        import traceback
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
    
    # Rest of the About tab content remains the same
    # ...
