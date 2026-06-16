"""Temp runner: run bench_fork at 720p and print the isolated transformer wall."""
import sys
import bench_fork
import model.propainter as P

if __name__ == '__main__':
    bench_fork.main()
    print("\n##### ISOLATED TRANSFORMER (CUDA-synced, no nsys) #####")
    print(f"PROPAINTER_FAST={__import__('os').environ.get('PROPAINTER_FAST','1')}")
    print(f"calls             : {P.TST_CALLS}")
    print(f"encoder    total  : {P.ENC_TIME_MS:.1f} ms  ({P.ENC_TIME_MS / max(P.TST_CALLS,1):.2f} ms/call)")
    print(f"transformer total : {P.TST_TIME_MS:.1f} ms  ({P.TST_TIME_MS / max(P.TST_CALLS,1):.2f} ms/call)")
    print("######################################################")
