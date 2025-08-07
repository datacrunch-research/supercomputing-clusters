import os
import subprocess
import json
import getpass
import socket
import re

# Helper to run a shell command and return output
def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return str(e)

# node_info
hostname = socket.gethostname()
user = getpass.getuser()
port = "22"  # default SSH port

print(f"Hostname: {hostname}")
print(f"User: {user}")
print(f"Port: {port}")


# Ubuntu info
lsb = run_cmd('lsb_release -a')
ubuntu = {
    'distributor id': '',
    'description': '',
    'release': '',
    'codename': ''
}
for line in lsb.splitlines():
    if ':' in line:
        k, v = line.split(':', 1)
        k = k.strip().lower()
        v = v.strip()
        if k == 'distributor id':
            ubuntu['distributor id'] = v.lower()
        elif k == 'description':
            ubuntu['description'] = v.lower()
        elif k == 'release':
            ubuntu['release'] = v
        elif k == 'codename':
            ubuntu['codename'] = v.lower()

print(f"Ubuntu: {ubuntu}")


# InfiniBand HCAs (using ibdev2netdev if available)
ib_hcas = {}
ibdev2netdev_out = run_cmd('ibdev2netdev')
if 'not found' not in ibdev2netdev_out and ibdev2netdev_out.strip():
    for line in ibdev2netdev_out.splitlines():
        # Example line: mlx5_bond_0 port 1 ==> bond0 (Up)
        parts = line.strip().split()
        if len(parts) >= 6 and parts[1] == 'port' and parts[3] == '==>':
            hca = parts[0]
            netdev = parts[4]
            ib_hcas[hca] = netdev
else:
    # Fallback: try ibstat -l
    ibstat = run_cmd('ibstat -l')
    if 'not found' not in ibstat:
        for hca in ibstat.split():
            port_name = run_cmd(f'ibstat {hca} | grep "CA type" || echo N/A')
            ib_hcas[hca] = port_name

print(f"ib_hcas: {ib_hcas}")

def parse_ibstat(output):
    adapters = []
    current_ca = None

    for line in output.splitlines():
        line = line.strip()

        if line.startswith("CA '"):
            if current_ca:
                adapters.append(current_ca)
            current_ca = {"name": re.search(r"CA '(.+)'", line).group(1)}

        elif current_ca is not None:
            if line.startswith("CA type:"):
                current_ca["ca_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Firmware version:"):
                current_ca["firmware_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("Hardware version:"):
                current_ca["hardware_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("Node GUID:"):
                current_ca["node_guid"] = line.split(":", 1)[1].strip()
            elif line.startswith("System image GUID:"):
                current_ca["system_image_guid"] = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                current_ca["port_state"] = line.split(":", 1)[1].strip()
            elif line.startswith("Physical state:"):
                current_ca["physical_state"] = line.split(":", 1)[1].strip()
            elif line.startswith("Rate:"):
                current_ca["rate"] = line.split(":", 1)[1].strip()
            elif line.startswith("Link layer:"):
                current_ca["link_layer"] = line.split(":", 1)[1].strip()

    # Append the last adapter
    if current_ca:
        adapters.append(current_ca)

    return adapters

devices_info = parse_ibstat(run_cmd('ibstat'))

print(devices_info)

print(f"devices_info: {devices_info}")

# Infiniband error info (best effort)
infiniband_error = {
    'driver_versions': {},
    'hcas': {},
    'error_counters': {}
}
# Try to get driver version
ofed_ver = run_cmd('ofed_info -s || echo N/A')
kernel_ver = run_cmd('uname -r || echo N/A')
driver_version = ofed_ver + " " + kernel_ver
firmware_ver = "v"+run_cmd('ibv_devinfo | grep "fw_ver" | head -1 | awk \'{print $2}\' || echo N/A')
infiniband_error['driver_versions']['Host Driver Version'] = driver_version
infiniband_error['driver_versions']['Firmware on CA'] = firmware_ver
infiniband_error['driver_versions']['total_passes'] = 0  # Placeholder

# Check with Johan
infiniband_error['hcas']['Port State'] = run_cmd('ibstat | grep "State:" | head -1 | awk \'{print $2}\' || echo N/A')
infiniband_error['hcas']['Node GUID'] = run_cmd('ibstat | grep "Node GUID" | head -1 | awk \'{print $3}\' || echo N/A')
infiniband_error['hcas']['total_passes'] = 0  # Placeholder
infiniband_error['error_counters']['total_passes'] = 0  # Placeholder
print(f"driver_version: {driver_version}")
print(f"firmware_ver: {firmware_ver}")


# VBIOS and CUDA/driver versions
nvidia_smi_out = run_cmd('nvidia-smi | grep "CUDA Version"')
nvidia_smi_parts = nvidia_smi_out.split()
Driver_Version = nvidia_smi_parts[5]


GSP_Firmware_Version = run_cmd('nvidia-smi -q | grep "GSP Firmware Version" | awk -F":" \'{print $2}\' | xargs -n1')
GSP_Firmware_Version = GSP_Firmware_Version.splitlines()[0] if GSP_Firmware_Version else ""
print(f"GSP_Firmware_Version: {GSP_Firmware_Version}")

Image_Version = run_cmd('nvidia-smi -q | grep "Image Version" | awk -F":" \'{print $2}\' | xargs -n1')
Image_Version = Image_Version.splitlines()[0] if Image_Version else ""
print(f"Image_Version: {Image_Version}")

Inforom_Version = run_cmd('nvidia-smi -q | grep "Inforom Version" | awk -F":" \'{print $2}\' | xargs -n1')
Inforom_Version = Inforom_Version.splitlines()[0] if Inforom_Version else ""
print(f"Inforom_Version: {Inforom_Version}")

VBIOS_Version = run_cmd('nvidia-smi -q | grep "VBIOS Version" | awk -F":" \'{print $2}\' | xargs -n1')
VBIOS_Version = VBIOS_Version.splitlines()[0] if VBIOS_Version else ""
print(f"VBIOS_Version: {VBIOS_Version}")

vbios = {
    'CUDA Version': run_cmd('nvcc --version | grep "release" | awk \'{print $5}\'').replace(',', ''),
    'Driver Version': Driver_Version,
    'GSP Firmware Version': GSP_Firmware_Version,
    'Image Version': Image_Version,
    'Inforom Version': Inforom_Version,
    'VBIOS Version': VBIOS_Version
}
print(f"vbios: {vbios}")

# NVLink information
def parse_nvlink_status(nvlink_output):
    """Parse nvidia-smi nvlink --status output similar to health_checks.py"""
    nvlink_info = {
        'gpu_count': 0,
        'gpu_model': '',
        'links_per_gpu': 0,
        'link_speed_gb_s': 0.0,
        'total_nvlink_bandwidth_per_gpu': 0.0,
        'links_status': {}
    }
    
    current_gpu = None
    gpu_models = set()
    link_speeds = []
    
    for line in nvlink_output.splitlines():
        line = line.strip()
        if line.startswith("GPU"):
            # Extract GPU info: "GPU 0: NVIDIA H200 (UUID: ...)"
            parts = line.split(": ")
            if len(parts) >= 2:
                gpu_id = parts[0]  # "GPU 0"
                gpu_model = parts[1].split(" (")[0]  # "NVIDIA H200"
                gpu_models.add(gpu_model)
                current_gpu = gpu_id
                nvlink_info['links_status'][current_gpu] = {}
                
        elif line.startswith("Link") and current_gpu:
            # Extract link info: "Link 0: 26.562 GB/s"
            try:
                parts = line.split(": ")
                link_id = parts[0].strip()  # "Link 0"
                speed_str = parts[1].strip()  # "26.562 GB/s"
                speed_val = float(speed_str.split()[0])  # 26.562
                
                nvlink_info['links_status'][current_gpu][link_id] = speed_val
                link_speeds.append(speed_val)
            except (ValueError, IndexError):
                continue
    
    # Calculate derived values
    nvlink_info['gpu_count'] = len(nvlink_info['links_status'])
    nvlink_info['gpu_model'] = list(gpu_models)[0] if gpu_models else 'Unknown'
    
    if nvlink_info['gpu_count'] > 0:
        links_per_gpu = len(nvlink_info['links_status'][list(nvlink_info['links_status'].keys())[0]])
        nvlink_info['links_per_gpu'] = links_per_gpu
        
        if link_speeds:
            nvlink_info['link_speed_gb_s'] = max(link_speeds)  # Use max speed found
            nvlink_info['total_nvlink_bandwidth_per_gpu'] = links_per_gpu * nvlink_info['link_speed_gb_s']
    
    return nvlink_info

# Get NVLink status
nvlink_output = run_cmd('nvidia-smi nvlink --status')
nvlink_info = parse_nvlink_status(nvlink_output)

# Get topology matrix
topology_output = run_cmd('nvidia-smi topo -m')
nvlink_info['topology_matrix'] = topology_output

print(f"NVLink info: {nvlink_info}")

# Docker info
docker_info = run_cmd('docker info --format "{{json .}}"')
docker = {
    'expected_error_info': {
        'Driver': ''
    },
    'expected_warning_info': {
        'CgroupVersion': '',
        'KernelVersion': '',
        'OSVersion': '',
        'ServerVersion': '',
        'ClientInfo': {
            'Version': '',
            'GoVersion': ''
        }
    }
}
try:
    import json as _json
    d = _json.loads(docker_info)
    docker['expected_error_info']['Driver'] = d.get('Driver', '')
    docker['expected_warning_info']['CgroupVersion'] = d.get('CgroupVersion', '')
    docker['expected_warning_info']['KernelVersion'] = d.get('KernelVersion', '')
    docker['expected_warning_info']['OSVersion'] = d.get('OperatingSystem', '')
    docker['expected_warning_info']['ServerVersion'] = d.get('ServerVersion', '')
    docker['expected_warning_info']['ClientInfo']['Version'] = d.get('ClientInfo', {}).get('Version', '')
    docker['expected_warning_info']['ClientInfo']['GoVersion'] = d.get('ClientInfo', {}).get('GoVersion', '')
except Exception:
    pass

print(f"docker: {docker}")

# Flint info (best effort) #TODO: HARDCODED INTERFACE mlx5_bond_0
flint_out = run_cmd('sudo flint -d mlx5_bond_0  q')
expected_fields = {}

flint_out = flint_out.splitlines()
for line in flint_out:
    if 'Image type' in line:
        expected_fields['Image type'] = line.split(':',1)[1].strip()
    elif 'FW Version' in line:
        expected_fields['FW Version'] = line.split(':',1)[1].strip()
    elif 'Product Version' in line:
        expected_fields['Product Version'] = line.split(':',1)[1].strip()
    elif 'FW Factory Version' in line:
        expected_fields['FW_Factory Version'] = line.split(':',1)[1].strip()
    elif 'Rom Info' in line:
        expected_fields['Rom Info'] = line.split(':',1)[1].strip()

flint = {
    'expected_fields': expected_fields,
    # TODO: Add expected_FW_possibilities 
    'expected_FW_possibilities': {}
}

print(f"flint: {flint}")


# Infiniband status (best effort)
infiniband_status = {
    'network_type': '',
    'active_devices': '',
    'device_names': []
}

rate_line = run_cmd("ibstat | grep 'Rate:' | head -1")

def classify_network_type(adapters):
    for adapter in adapters:
        rate_line = adapter.get("rate", "")
        link_layer = adapter.get("link_layer", "")

        if link_layer == "InfiniBand":
            if "400" in rate_line:
                adapter["network_type"] = "NDR"
            elif "200" in rate_line:
                adapter["network_type"] = "HDR"
            elif "100" in rate_line:
                adapter["network_type"] = "EDR"
            elif "56" in rate_line:
                adapter["network_type"] = "FDR"
            elif "40" in rate_line:
                adapter["network_type"] = "QDR"
            elif "20" in rate_line:
                adapter["network_type"] = "DDR"
            elif "10" in rate_line:
                adapter["network_type"] = "SDR"
            else:
                adapter["network_type"] = "Unknown InfiniBand"
        elif link_layer == "Ethernet":
            adapter["network_type"] = "Ethernet"
        else:
            adapter["network_type"] = "Unknown"

    return adapters

# Care, if there are no infiniband adapters, the function will return an Ethernet as network type and 0 active devices with empty list of device names
classified_adapters = classify_network_type(devices_info)

ib_adapters = [a['name'] for a in devices_info if a.get('link_layer') == 'InfiniBand' and a.get('port_state') == 'Active']
infiniband_status['active_devices'] = len(ib_adapters)
infiniband_status['network_type'] = list( adapter['network_type'] for adapter in classified_adapters)
infiniband_status['device_names'] = ib_adapters

print(f"infiniband_status: {infiniband_status}")
# Compose the config dict
config = {
    'node_info': {
        'nodes': [hostname],
        'port': port,
        'user': user
    },
    'ubuntu': ubuntu,
    'ib_hcas': ib_hcas,
    'infiniband_error': infiniband_error,
    'vbios': vbios,
    'nvlink': nvlink_info,
    'docker': docker,
    'flint': flint,
    'infiniband_status': infiniband_status
}

with open('generated_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("System config written to generated_config.json.") 
