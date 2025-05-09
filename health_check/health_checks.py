# A simplified Infiniband health check script with modular testing
# Provides a general IBHealthCheck class that performs multiple IB-related checks

import subprocess
import re
from typing import List, Dict, Optional


class Port:
    def __init__(self, port_number: int, properties: Dict[str, str]):
        self.port_number = port_number
        self.state = properties.get("State")
        self.physical_state = properties.get("Physical state")
        self.rate = properties.get("Rate")
        self.base_lid = properties.get("Base lid")
        self.lmc = properties.get("LMC")
        self.sm_lid = properties.get("SM lid")
        self.capability_mask = properties.get("Capability mask")
        self.port_guid = properties.get("Port GUID")
        self.link_layer = properties.get("Link layer")

    def __repr__(self):
        return f"Port({self.port_number}, {self.__dict__})"

class ChannelAdapter:
    def __init__(self, name: str):
        self.name = name
        self.ca_type = None
        self.num_ports = None
        self.fw_version = None
        self.hw_version = None
        self.node_guid = None
        self.system_image_guid = None
        self.ports: List[Port] = []

    def __repr__(self):
        return f"CA({self.name}, {self.__dict__})"

    def add_port(self, port: Port):
        self.ports.append(port)

class IBStatParser:
    def __init__(self, ibstat_output: str):
        self.ibstat_output = ibstat_output
        self.ca_list: List[ChannelAdapter] = []

    def parse(self):
        lines = self.ibstat_output.splitlines()
        current_ca = None
        current_port_number = None
        port_props = {}

        for line in lines:
            line = line.strip()
            ca_match = re.match(r"CA '(.+)'", line)
            if ca_match:
                if current_ca:
                    if port_props:
                        current_ca.add_port(Port(current_port_number, port_props))
                        port_props = {}
                        current_port_number = None
                    self.ca_list.append(current_ca)
                current_ca = ChannelAdapter(ca_match.group(1))
                continue
            
            if current_ca is not None:
                key_val = re.match(r"([^:]+):\s+(.+)", line)
                if key_val:
                    key, val = key_val.groups()
                    key = key.strip()
                    val = val.strip()
                    if current_port_number is not None:
                        port_props[key] = val
                    else:
                        # Top-level CA info
                        if key == "CA type":
                            current_ca.ca_type = val
                        elif key == "Number of ports":
                            current_ca.num_ports = int(val)
                        elif key == "Firmware version":
                            current_ca.fw_version = val
                        elif key == "Hardware version":
                            current_ca.hw_version = val
                        elif key == "Node GUID":
                            current_ca.node_guid = val
                        elif key == "System image GUID":
                            current_ca.system_image_guid = val

                elif re.match(r"Port\s+(\d+)", line):
                    if port_props:
                        current_ca.add_port(Port(current_port_number, port_props))
                    current_port_number = int(line.split()[1].replace(":", ""))
                    port_props = {}

        if current_ca:
            if port_props:
                current_ca.add_port(Port(current_port_number, port_props))
            self.ca_list.append(current_ca)

    def get_channel_adapters(self) -> List[ChannelAdapter]:
        return self.ca_list

class PCIEStatParser:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.device_desc: Optional[str] = None
        self.numa_node: Optional[int] = None
        self.lnkcap_speed: Optional[str] = None
        self.lnkcap_width: Optional[str] = None
        self.lnksta_speed: Optional[str] = None
        self.lnksta_width: Optional[str] = None
        self.kernel_driver: Optional[str] = None
        self.kernel_modules: Optional[str] = None
        self.error: Optional[str] = None

    def __repr__(self):
        return f"PCIEInfo({self.__dict__})"

class PCIEInfoParser:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.info = PCIEStatParser(device_id)

    def parse(self):
        try:
            result = subprocess.run(
                ["sudo","lspci", "-s", self.device_id, "-vv"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            output = result.stdout
        except subprocess.CalledProcessError as e:
            self.info.error = f"Failed to run lspci: {e.stderr}"
            return

        for line in output.splitlines():
            line = line.strip()
            if "Ethernet controller:" in line:
                self.info.device_desc = line
            elif "NUMA node:" in line:
                match = re.search(r"NUMA node: (\d+)", line)
                if match:
                    self.info.numa_node = int(match.group(1))
            elif line.startswith("LnkCap:"):
                match = re.search(r"Speed (\d+GT/s), Width x(\d+)", line)
                if match:
                    self.info.lnkcap_speed = match.group(1)
                    self.info.lnkcap_width = f"x{match.group(2)}"
            elif line.startswith("LnkSta:"):
                match = re.search(r"Speed (\d+GT/s).*Width x(\d+)", line)
                if match:
                    self.info.lnksta_speed = match.group(1)
                    self.info.lnksta_width = f"x{match.group(2)}"
            elif "Kernel driver in use:" in line:
                self.info.kernel_driver = line.split(":", 1)[1].strip()
            elif "Kernel modules:" in line:
                self.info.kernel_modules = line.split(":", 1)[1].strip()

    def get_info(self) -> PCIEStatParser:
        return self.info

class IBHealthCheck:
    def __init__(self):
        self.results: Dict[str, str] = {}

    def run_command(self, cmd: str) -> str:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()

    # Detects hardware PCIe link degradations on NICs
    def check_pcie_info(self, device_id: str = "d3:00.1") -> None:
        print("Checking PCIe link configuration")
        parser = PCIEInfoParser(device_id)
        parser.parse()
        info = parser.get_info()
        if info.lnkcap_speed == info.lnksta_speed and info.lnkcap_width == info.lnksta_width:
            self.results[f"PCIe {device_id} device cap limit speed and width"] = "OK"
        else:
            self.results[f"{device_id}_speed"] = "FAIL"

        if info.error:
            self.results[f"{device_id}_error"] = info.error

    # Validates the basic health of Mellanox NICs/Infiniband Host Channel Adapters 
    def check_hca_self_test(self, command:str) -> None:
        print("Checking HCA self test")
        output = self.run_command(command)

        firmware_fail = any("Firmware Check" in line and "FAIL" in line for line in output.splitlines())
        self.results["firmware_status"] = "FAIL" if firmware_fail else "OK"

        guid_lines = [line for line in output.splitlines() if "Node GUID" in line and ("NA" in line or "00:00" in line)]
        self.results["node_guids"] = "FAIL" if guid_lines else "OK"

        match = re.search(r"Number of CA Ports Active.*?(\d+)", output)
        if not match or int(match.group(1)) < 1:
            self.results["active_ports"] = "FAIL"
        else:
            self.results["active_ports"] = "OK"

        match = re.search(r"Host Driver Version\s+\.\.+\s+(.*)", output)
        self.results["driver_version"] = match.group(1).strip() if match else "Unavailable"

    # Verifies link state and type of each HCA port.
    def check_ibstatus(self,command:str, expected_active_count: int = 8) -> None:
        print("Checking InfiniBand status")
        output = self.run_command(command)
        try:
            active_count = int(output.strip())
            self.results["ibstatus_active"] = "OK" if active_count == expected_active_count else f"FAIL ({active_count}/{expected_active_count})"
        except ValueError:
            self.results["ibstatus_active"] = "FAIL (unreadable output)"

    # Checks the consistency of NVLink link speeds between GPUs
    def check_nvlink_speed_consistency(self, nvlink_output: str) -> None:
        print("Checking NVLink link speeds...")
        nvlink_output = nvlink_output
        current_gpu = None
        expected_speed = None
        inconsistent_links = {}

        for line in nvlink_output.splitlines():
            line = line.strip()
            if line.startswith("GPU"):
                current_gpu = line.split(":")[0]
            elif line.startswith("Link") and current_gpu:
                try:
                    parts = line.split(":")
                    link_id = parts[0].strip()       # e.g., "Link 0"
                    speed_val = float(parts[1].strip().split()[0])  # e.g., "26.562"
                    
                    if expected_speed is None:
                        expected_speed = speed_val

                    if abs(speed_val - expected_speed) > 0.01:
                        if current_gpu not in inconsistent_links:
                            inconsistent_links[current_gpu] = []
                        inconsistent_links[current_gpu].append(f"{link_id}: {speed_val} GB/s")
                except Exception as e:
                    self.results["nvlink_speed"] = f"FAIL: Could not parse speed line '{line}'"
                    return

        if inconsistent_links:
            detail_lines = [
                f"{gpu} -> {', '.join(link_infos)}" for gpu, link_infos in inconsistent_links.items()
            ]
            self.results["nvlink_speed"] = f"FAIL: inconsistent link speeds\n" + "\n".join(detail_lines)
        else:
            self.results["nvlink_speed"] = "OK"

    def net_topology(self) -> None:
            print("Checking network topology")
            cmd = "nvidia-smi topo -m"
            output = self.run_command(cmd)
            self.results["Net topology"] = output

    def run_all(self) -> Dict[str, str]:
        self.check_hca_self_test(self.run_command("sudo flock -w 90 -F /usr/bin/hca_self_test.ofed /usr/bin/hca_self_test.ofed </dev/null"))
        self.check_ibstatus(self.run_command("ibstatus | grep -B 2 'InfiniBand' | grep -c ACTIVE"))
        self.check_pcie_info()
        #self.net_topology()
        self.check_nvlink_speed_consistency(self.run_command("nvidia-smi nvlink --status"))
        self.check_nvlink_errors(self.run_command("nvidia-smi nvlink --errorcounters"))
        return self.results

    def test_IBStatParser():
        test="""
        CA 'mlx5_0'
        CA type: MT4129
        Number of ports: 1
        Firmware version: 28.42.1224
        Hardware version: 0
        Node GUID: 0xc470bd0300edcfd8
        System image GUID: 0xc470bd0300edcfd8
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 10
            Base lid: 65535
            LMC: 0
            SM lid: 0
            Capability mask: 0xa751e848
            Port GUID: 0xc470bd0300edcfd8
            Link layer: InfiniBand
        Port 2:
            State: Down
            Physical state: Disabled
            Rate: 10
            Base lid: 65535
            LMC: 0
            SM lid: 0
            Capability mask: 0xa751e848
            Port GUID: 0xc470bd0300edcfd8
            Link layer: InfiniBand

        CA 'mlx5_1'
        CA type: MT4129
        Number of ports: 1
        Firmware version: 28.42.1224
        Hardware version: 0
        Node GUID: 0xc470bd0300edcfdc
        System image GUID: 0xc470bd0300edcfdc
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 10
            Base lid: 65535
            LMC: 0
            SM lid: 0
            Capability mask: 0xa751e848
            Port GUID: 0xc470bd0300edcfdc
            Link layer: InfiniBand

        CA 'mlx5_2'
        CA type: MT41692
        Number of ports: 1
        Firmware version: 32.42.1000
        Hardware version: 1
        Node GUID: 0xc470bd030093b1e2
        System image GUID: 0xc470bd030093b1e2
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 40
            Base lid: 0
            LMC: 0
            SM lid: 0
            Capability mask: 0x00010000
            Port GUID: 0xc670bdfffe93b1e2
            Link layer: Ethernet

        CA 'mlx5_3'
        CA type: MT41692
        Number of ports: 1
        Firmware version: 32.42.1000
        Hardware version: 1
        Node GUID: 0xc470bd030093b1e3
        System image GUID: 0xc470bd030093b1e2
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 40
            Base lid: 0
            LMC: 0
            SM lid: 0
            Capability mask: 0x00010000
            Port GUID: 0xc670bdfffe93b1e3
            Link layer: Ethernet

        CA 'mlx5_4'
        CA type: MT4129
        Number of ports: 1
        Firmware version: 28.42.1224
        Hardware version: 0
        Node GUID: 0xc470bd0300edd0a0
        System image GUID: 0xc470bd0300edd0a0
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 10
            Base lid: 65535
            LMC: 0
            SM lid: 0
            Capability mask: 0xa751e848
            Port GUID: 0xc470bd0300edd0a0
            Link layer: InfiniBand

        CA 'mlx5_5'
        CA type: MT4129
        Number of ports: 1
        Firmware version: 28.42.1224
        Hardware version: 0
        Node GUID: 0xc470bd0300edd0a4
        System image GUID: 0xc470bd0300edd0a4
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 10
            Base lid: 65535
            LMC: 0
            SM lid: 0
            Capability mask: 0xa751e848
            Port GUID: 0xc470bd0300edd0a4
            Link layer: InfiniBand

        CA 'mlx5_6'
        CA type: MT41692
        Number of ports: 1
        Firmware version: 32.42.1000
        Hardware version: 1
        Node GUID: 0xc470bd030093b196
        System image GUID: 0xc470bd030093b196
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 40
            Base lid: 0
            LMC: 0
            SM lid: 0
            Capability mask: 0x00010000
            Port GUID: 0xc670bdfffe93b196
            Link layer: Ethernet

        CA 'mlx5_7'
        CA type: MT41692
        Number of ports: 1
        Firmware version: 32.42.1000
        Hardware version: 1
        Node GUID: 0xc470bd030093b197
        System image GUID: 0xc470bd030093b196
        Port 1:
            State: Down
            Physical state: Disabled
            Rate: 40
            Base lid: 0
            LMC: 0
            SM lid: 0
            Capability mask: 0x00010000
            Port GUID: 0xc670bdfffe93b197
            Link layer: Ethernet
        """
        ibstat_parser = IBStatParser(test.strip())
        ibstat_parser.parse()
        devices_info = ibstat_parser.get_channel_adapters()
        for device in devices_info:
            print(f"{device} \n")

    def test_check_nvlink_speed_consistency():
        test_fail = """
GPU 0: NVIDIA H200 (UUID: GPU-8f0500a5-abae-41df-22f9-42f86172866c)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 1.0 GB/s
         Link 3: 1.562 GB/s
         Link 4: 1.562 GB/s
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
         Link 8: 26.562 GB/s
         Link 9: 26.562 GB/s
         Link 10: 1.562 GB/s
         Link 11: 26.562 GB/s
         Link 12: 26.562 GB/s
         Link 13: 26.562 GB/s
         Link 14: 26.562 GB/s
         Link 15: 26.562 GB/s
         Link 16: 26.562 GB/s
         Link 17: 26.562 GB/s
GPU 1: NVIDIA H200 (UUID: GPU-9e8682f0-0d01-e118-3a86-008e972bfe2a)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 26.562 GB/s
         Link 3: 26.562 GB/s
         Link 4: 26.562 GB/s
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
         Link 8: 26.562 GB/s
         Link 9: 26.562 GB/s
         Link 10: 26.562 GB/s
         Link 11: 26.562 GB/s
         Link 12: 26.562 GB/s
         Link 13: 26.562 GB/s
         Link 14: 26.562 GB/s
         Link 15: 26.562 GB/s
         Link 16: 26.562 GB/s
         Link 17: 26.562 GB/s
"""
        checker = IBHealthCheck()
        checker.check_nvlink_speed_consistency(test_fail)
        print(checker.results["nvlink_speed"])

    def check_nvlink_errors(self, nvlink_err_output: str) -> None:
        print("Checking NVLink error counters...")

        current_gpu = None
        errors_detected = {}

        for line in nvlink_err_output.splitlines():
            line = line.strip()
            if line.startswith("GPU"):
                current_gpu = line.split(":")[0]
            elif line.startswith("Link") and current_gpu:
                link_match = re.match(r"Link (\d+): .*", line)
                if not link_match:
                    continue
                link_id = f"Link {link_match.group(1)}"
                counters = re.findall(r"(\w+ Error Counter): (\d+)", line)
                for counter_name, value in counters:
                    if int(value) > 0:
                        if current_gpu not in errors_detected:
                            errors_detected[current_gpu] = []
                        errors_detected[current_gpu].append(f"{link_id} - {counter_name} = {value}")

        if errors_detected:
            lines = [f"{gpu}: {', '.join(errors)}" for gpu, errors in errors_detected.items()]
            self.results["nvlink_errors"] = "FAIL:\n" + "\n".join(lines)
        else:
            self.results["nvlink_errors"] = "OK"

if __name__ == "__main__":
    
    checker = IBHealthCheck()
    results = checker.run_all()
    print("\nInfiniband Health Check Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
