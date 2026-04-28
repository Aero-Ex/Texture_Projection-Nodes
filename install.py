import os
import sys
import subprocess
import platform

def get_cuda_ver():
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda.replace('.', '')
            return f"cu{cuda_ver}"
    except ImportError:
        pass
    return None

def get_torch_ver():
    try:
        import torch
        full_ver = torch.__version__.split('+')[0]
        # Map 2.4.0 -> torch24, 2.10.1 -> torch210
        parts = full_ver.split('.')
        if len(parts) >= 2:
            return f"torch{parts[0]}{parts[1]}"
    except ImportError:
        pass
    return None

def get_py_ver():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"

def get_platform_tag():
    p = platform.system().lower()
    if p == "windows":
        return "win_amd64"
    elif p == "linux":
        return "manylinux_2_34_x86_64.manylinux_2_35_x86_64"
    return None

def install_requirements():
    print("Installing base requirements from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def find_local_wheel(name, cuda, torch_v, py, plat):
    wheels_dir = os.path.join(os.path.dirname(__file__), "wheels")
    if not os.path.exists(wheels_dir):
        return None
    for f in os.listdir(wheels_dir):
        if not f.endswith(".whl"): continue
        f_lower = f.lower()
        if name.replace("-", "_") in f_lower or name in f_lower:
            # Check Python version e.g. cp312
            if py not in f_lower: continue
            # Check platform e.g. win_amd64
            if plat not in f_lower: continue
            cuda_num = cuda.replace("cu", "")
            if cuda_num not in f_lower and f"cuda{cuda_num}" not in f_lower and cuda not in f_lower: 
                continue
            
            # Torch match
            torch_num = torch_v.replace("torch", "")
            if torch_num not in f_lower.replace(".", ""):
                continue
                
            return os.path.join(wheels_dir, f)
    return None

def install_cuda_wheels():
    cuda = get_cuda_ver()
    torch_v = get_torch_ver()
    py = get_py_ver()
    plat = get_platform_tag()

    if not all([cuda, torch_v, py, plat]):
        print(f"Warning: Could not detect all environment variables (CUDA: {cuda}, Torch: {torch_v}, Python: {py}, Platform: {plat}).")
        return

    print(f"Detected Environment: CUDA={cuda}, Torch={torch_v}, Python={py}, Platform={plat}")

    # Wheel versions for remote fallback
    nvdiffrast_ver = "0.4.0"
    nvdiffrec_render_ver = "0.0.1"

    packages = ["nvdiffrast", "nvdiffrec-render", "custom-rasterizer"]
    
    for name in packages:
        # 1. Try local wheel first
        local_wheel = find_local_wheel(name, cuda, torch_v, py, plat)
        if local_wheel:
            print(f"Found local wheel for {name}: {local_wheel}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", local_wheel])
                print(f"Successfully installed {name} from local wheel.")
                continue
            except subprocess.CalledProcessError:
                print(f"Failed to install local wheel {local_wheel}. Trying remote/manual...")

        # 2. Remote fallback for specific packages
        if name == "nvdiffrast":
            url = f"https://pozzettiandrea.github.io/cuda-wheels/nvdiffrast/nvdiffrast-{nvdiffrast_ver}+{cuda}{torch_v}-{py}-{py}-{plat}.whl"
        elif name == "nvdiffrec-render":
            url = f"https://pozzettiandrea.github.io/cuda-wheels/nvdiffrec-render/nvdiffrec_render-{nvdiffrec_render_ver}+{cuda}{torch_v}-{py}-{py}-{plat}.whl"
        else:
            print(f"No local wheel for {name} and no remote source defined. Skipping automated install.")
            continue

        print(f"Attempting to install {name} from remote wheel: {url}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", url])
            print(f"Successfully installed {name}.")
        except subprocess.CalledProcessError:
            print(f"Failed to install {name} via remote wheel.")
            if name == "custom-rasterizer":
                print("Manual build required: cd Texture_Projection/Renderer/custom_rasterizer && python setup.py install")

def test_all_configs():
    print("Running Dry Run for multiple system configurations (Remote & Local)...")
    
    test_cases = [
        {"cuda": "12.4", "torch": "2.4.0", "py": "3.10", "os": "linux"},
        {"cuda": "13.0", "torch": "2.9.1", "py": "3.11", "os": "windows"},
        {"cuda": "12.6", "torch": "2.6.0", "py": "3.12", "os": "windows"},
        {"cuda": "13.0", "torch": "2.10.0", "py": "3.12", "os": "windows"},
    ]

    for case in test_cases:
        cuda = f"cu{case['cuda'].replace('.', '')}"
        torch_parts = case['torch'].split('.')
        torch_v = f"torch{torch_parts[0]}{torch_parts[1]}"
        py = f"cp{case['py'].replace('.', '')}"
        plat = "win_amd64" if case['os'] == "windows" else "manylinux_2_34_x86_64.manylinux_2_35_x86_64"

        print(f"\n--- Testing: {case['os']} | Py{case['py']} | Torch{case['torch']} | CUDA{case['cuda']} ---")
        
        for name in ["nvdiffrast", "nvdiffrec-render", "custom-rasterizer"]:
            local = find_local_wheel(name, cuda, torch_v, py, plat)
            if local:
                print(f"{name} [LOCAL]: {os.path.basename(local)}")
            else:
                if name != "custom-rasterizer":
                    # Construct remote URL for demo
                    prefix = "nvdiffrast-0.4.0" if name == "nvdiffrast" else "nvdiffrec_render-0.0.1"
                    url = f"https://pozzettiandrea.github.io/cuda-wheels/{name}/{prefix}+{cuda}{torch_v}-{py}-{py}-{plat}.whl"
                    print(f"{name} [REMOTE]: {url}")
                else:
                    print(f"{name} [NO MATCH FOUND]")

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        test_all_configs()
        sys.exit(0)

    # Ensure torch is installed first to detect CUDA/Torch version
    print("Pre-installing torch for version detection...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    except:
        print("Warning: Could not pre-install torch. Detection might fail.")
    
    if install_requirements():
        install_cuda_wheels()
    
    print("\nInstallation complete. Please remember to build the custom_rasterizer if needed:")
    print("cd Texture_Projection/Renderer/custom_rasterizer && python setup.py install")
