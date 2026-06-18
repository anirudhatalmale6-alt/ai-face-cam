import os
import sys
import ctypes

base = os.path.dirname(os.path.abspath(__file__))
torch_lib = os.path.join(base, 'venv', 'lib', 'site-packages', 'torch', 'lib')

print("=== FasterLivePortrait DLL Fix ===")
print(f"Torch lib: {torch_lib}")
print()

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(torch_lib)
    print("Added torch lib to DLL search path")

os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')

try:
    import torch
    print(f"PyTorch loaded OK! Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    print("Starting camera mode...")
    print()

    sys.argv = [
        'run.py',
        '--src_image', 'assets\\examples\\source\\s12.jpg',
        '--driving_type', 'camera',
    ]
    os.chdir(base)
    exec(open(os.path.join(base, 'run.py')).read())

except Exception as e:
    print(f"PyTorch failed to load: {e}")
    print()
    print("Diagnosing which DLLs fail...")
    print()

    failed = []
    ok = []
    for dll_name in sorted(os.listdir(torch_lib)):
        if not dll_name.endswith('.dll'):
            continue
        dll_path = os.path.join(torch_lib, dll_name)
        try:
            ctypes.CDLL(dll_path)
            ok.append(dll_name)
        except Exception as e2:
            failed.append((dll_name, str(e2)))
            print(f"  FAIL: {dll_name}")
            print(f"        {e2}")
            print()

    print(f"\n{len(ok)} DLLs loaded OK, {len(failed)} failed")

    if failed:
        print("\nFailed DLLs:")
        for name, err in failed:
            print(f"  - {name}")
        print("\nPlease screenshot this and send to your developer.")

    input("\nPress Enter to exit...")
