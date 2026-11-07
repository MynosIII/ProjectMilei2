"""Run the analysis with all Python socket connections disabled.
Only scripts/refresh.py downloads the configured public Google CSV inputs.
"""
import os,runpy,socket,sys,sysconfig
from pathlib import Path

def offline(*args,**kwargs):
    raise PermissionError('Network access is disabled during survey calculation.')

OriginalSocket=socket.socket
class OfflineSocket(OriginalSocket):
    connect=offline
    connect_ex=offline
    sendto=offline
socket.socket=OfflineSocket
socket.create_connection=offline
socket.getaddrinfo=offline
# Isolated Python (-I -S) skips site initialization and .pth startup hooks.
# Add only the installed package directory after denying network connections.
sys.path.append(sysconfig.get_path('purelib'))
os.environ['DO_NOT_TRACK']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['WANDB_DISABLED']='true'
target=Path(__file__).parent/'pjm_analisis_avanzado.py'
sys.path.insert(0,str(target.parent))
runpy.run_path(str(target),run_name='__main__')
