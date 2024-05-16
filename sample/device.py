
import time
import sys
from sample.libdaq import *
def device_example(index):
    errorcode,oldname = libdaq_device_get_name(index)
    print("daq device name:",oldname)
 
    libdaq_device_rename_byindex(index, b'DAQ_newname1')
    print("daq device rename by index to DAQ_newname1 success!")

    libdaq_device_rename_byname(b'DAQ_newname1',b'DAQ_newname2')
    print("daq device rename by index to DAQ_newname2 success!")

	#restore name
    libdaq_device_rename_byindex(index, oldname)
    print("daq device restore to old name: %s success!" %(oldname))

    #UID test
    print("setUID_byindex: blight at 1Hz")
    for i in range(0,2):
        libdaq_device_setUID_byindex(index,UID_ON)
        time.sleep(0.5)
        libdaq_device_setUID_byindex(index,UID_OFF)
        time.sleep(0.5)

    print("setUID_byName: blight at 1Hz")
    for i in range(0,2):
        libdaq_device_setUID_byname(oldname,UID_ON)
        time.sleep(0.5)
        libdaq_device_setUID_byname(oldname,UID_OFF)
        time.sleep(0.5)

    #get version
    print("libqab and device version:")
    errorcode,libdaq_version=libdaq_get_version()
    errorcode,hw_version=libdaq_device_get_version(oldname)
    print("library version:%d.%d.%d" % (libdaq_version.major_ver,libdaq_version.minor_ver,libdaq_version.micro_ver))
    print("firmware version:%d.%d.%d" % (hw_version.firmware_major,hw_version.firmware_minor,hw_version.firmware_micro))
    print("pcb version:%d" % hw_version.pcb_ver)
    print("bom version:%d" % hw_version.bom_ver)