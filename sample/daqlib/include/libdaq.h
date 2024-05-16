/*
 * @file    libdaq.h
 * @author  libdaq Library Team
 * @version V2.3.1
 * @date    2020-10-17
 * @brief   This file provide all libdaq API functions and definitions.
 *
 * Copyright (c) 2016-2020 ZishuTech.Co.Ltd. All rights reserved.
 */

#ifndef DAQLIB_H
#define DAQLIB_H

#define LIBDAQ_API_VERSION 0x020301  //2.3.1

#ifdef __cplusplus
extern "C"{
#endif


#if defined(_WIN32) || defined(__CYGWIN__) || defined(_WIN32_WCE)

#if defined(DAQDLL_LIBRARY)
#  define DLL_API __declspec(dllexport)
#else
#  define DLL_API __declspec(dllimport)
#endif
#define DLL_CALL __cdecl

#else
#define DLL_API
#define DLL_CALL
#endif




enum daqlib_error_code {
    /** Success (no error) */
    LIBDAQ_SUCCESS = 0,

    /** Input/output error */
    LIBDAQ_ERROR_IO = -1,

    /** Invalid parameter */
    LIBDAQ_ERROR_INVALID_PARAM = -2,

    /** Access denied (insufficient permissions) */
    LIBDAQ_ERROR_ACCESS = -3,

    /** No such device (it may have been disconnected) */
    LIBDAQ_ERROR_NO_DEVICE = -4,

    /** Entity not found */
    LIBDAQ_ERROR_NOT_FOUND = -5,

    /** Resource busy */
    LIBDAQ_ERROR_BUSY = -6,

    /** Operation timed out */
    LIBDAQ_ERROR_TIMEOUT = -7,

    /** Overflow */
    LIBDAQ_ERROR_OVERFLOW = -8,

    /** Pipe error */
    LIBDAQ_ERROR_PIPE = -9,

    /** System call interrupted (perhaps due to signal) */
    LIBDAQ_ERROR_INTERRUPTED = -10,

    /** Insufficient memory */
    LIBDAQ_ERROR_NO_MEM = -11,

    /** Operation not supported or unimplemented on this platform */
    LIBDAQ_ERROR_NOT_SUPPORTED = -12,

    /* NB: Remember to update DAQLIB_ERROR_COUNT below as well as the
       message strings in strerror.c when adding new error codes here. */

    /** no such module*/
    LIBDAQ_ERROR_NO_MODULE = -13,

    /** module not match*/
    LIBDAQ_ERROR_MODULE_TYPE = -14,

    /** no such gpio pin*/
    LIBDAQ_ERROR_NO_GPIOPIN = -15,

    /** gpio pin with config error*/
    LIBDAQ_ERROR_GPIO_CONFIG = -16,

    /** gpio pin with index error*/
    LIBDAQ_ERROR_GPIO_INDEX = -17,

    /** operarion on such pin is not supported */
    LIBDAQ_ERROR_GPIOPIN_NOT_SUPPORTED = -18,

    /** deivce reply error */
    LIBDAQ_ERROR_REPLY = -19,

    /** device transfer status error*/
    LIBDAQ_ERROE_TRANSFER =-20,

    /** Other error */
    LIBDAQ_ERROR_OTHER = -99,
};

typedef struct hw_version {
    unsigned char firmware_major;	/** firmware major version. */
    unsigned char firmware_minor;	/** firmware minor version. */
    unsigned char firmware_micro;	/** firmware micro version. */

    unsigned char pcb_ver;	/** PCB  version. */
    unsigned char bom_ver;	/** BOM  version. */
}hw_version_t;

#define LIBDAQ_ACCESS_MODE_DEV_NAME_SN_BOTH 0x00  //defaut
#define LIBDAQ_ACCESS_MODE_DEV_NAME_ONLY    0x01
#define LIBDAQ_ACCESS_MODE_DEV_SN_ONLY      0x02

/*libdaq initiate API */
DLL_API int DLL_CALL libdaq_init(void);
DLL_API int DLL_CALL libdaq_exit(void);
DLL_API int DLL_CALL libdaq_scan_device(void);

DLL_API void DLL_CALL libdaq_set_option(int option);
/* get libdaq version */
DLL_API int DLL_CALL libdaq_get_version(char * major_ver,char * minor_ver,char * micro_ver);

/* Error relate API */
DLL_API  const char * DLL_CALL libdaq_get_error_desc(int error_code);
DLL_API  const char * DLL_CALL libdaq_get_error_str(int error_code);

/*
 * device manage struct and API definition
 */
DLL_API int DLL_CALL libdaq_device_get_version(const char * device_name,struct hw_version *hw_version);
DLL_API int DLL_CALL libdaq_device_get_count(void);
DLL_API int DLL_CALL libdaq_device_rename_byindex(unsigned int device_index,const char * newname );
DLL_API int DLL_CALL libdaq_device_rename_byname(const char * device_name,const char * newname);
DLL_API int DLL_CALL libdaq_device_get_name(unsigned int index, char* device_name, int length);
DLL_API int DLL_CALL libdaq_device_get_type(const char * device_name, char* device_type,int length);
DLL_API int DLL_CALL libdaq_device_setUID_byindex(unsigned int index,unsigned char state);//indicate by UID
DLL_API int DLL_CALL libdaq_device_setUID_byname(const char * device_name,unsigned char state); //indicate by UID
DLL_API int DLL_CALL libdaq_device_factory_reset(const char * device_name);
DLL_API int DLL_CALL libdaq_device_reboot(const char * device_name);

/*
 * GPIO module struct and API definition
 */
typedef enum {
    IOMODE_IN=0,     // input only
    IOMODE_OUT,      // output only
    IOMODE_IN_OUT,   // can be configured input or output
    IOMODE_IN_AF,    // can be configured input or Alternate function
    IOMODE_OUT_AF,   // can be configured output or Alternate function
    IOMODE_IN_OUT_AF // can be configured input, output or Alternate function
}iomode_t;

typedef enum {
    IOCONF_IN=0,
    IOCONF_OUT,
    IOCONF_AF
}ioconf_t;

typedef struct {
    iomode_t iomode;
}ioattr_t;

DLL_API int DLL_CALL libdaq_gpio_get_iocount(const char* device_name, const char *module_name, unsigned char *iocount);
DLL_API int DLL_CALL libdaq_gpio_get_ioattrs(const char* device_name, const char *module_name, ioattr_t *ioattr);
DLL_API int DLL_CALL libdaq_gpio_get_config(const char* device_name, const char *module_name, ioconf_t *confs);
DLL_API int DLL_CALL libdaq_gpio_set_config(const char* device_name,const char *module_name,ioconf_t *confs);
DLL_API int DLL_CALL libdaq_gpio_write_bit(const char* device_name, const char *module_name, unsigned char ioIndex, unsigned char BitVal);
DLL_API int DLL_CALL libdaq_gpio_write_port(const char* device_name,const char *module_name, unsigned char *PortVal);
DLL_API int DLL_CALL libdaq_gpio_read_bit(const char* device_name, const char *module_name, unsigned char ioIndex, unsigned char * BitVal);
DLL_API int DLL_CALL libdaq_gpio_read_port(const char* device_name,const char *module_name,unsigned char *PortVal);

/*
 * DAC and ADC module channel range definition
 */
#define CHANNEL_RANGE_0_P25mV     		(85)//range:0-25mV
#define CHANNEL_RANGE_0_P50mV     		(0)//range:0-50mV
#define CHANNEL_RANGE_0_P0V1     		(1)//range:0-0.1V
#define CHANNEL_RANGE_0_P0V25     		(2)//range:0-0.25V
#define CHANNEL_RANGE_0_P0V5     		(3)//range:0-0.5V 
#define CHANNEL_RANGE_0_P1V     		(4)//range:0-1V
#define CHANNEL_RANGE_0_P2V     		(8)//range:0-2V
#define CHANNEL_RANGE_0_P2V5    		(12)//range:0-2.5V
#define CHANNEL_RANGE_0_P5V     		(16)//range:0-5V
#define CHANNEL_RANGE_0_P10V    		(20)//range:0-10V
#define CHANNEL_RANGE_0_P12V    		(91)//range:0-12V
#define CHANNEL_RANGE_0_P12V5    		(92)//range:0-12.5V
#define CHANNEL_RANGE_0_P15V    		(21)//range:0-15V
#define CHANNEL_RANGE_N78mV125_P78mV125 (81)//range:+/-78.125mV
#define CHANNEL_RANGE_N156mV25_P156mV25 (82)//range:+/-156.25mV
#define CHANNEL_RANGE_N312mV5_P312mV5   (83)//range:+/-312.5mV
#define CHANNEL_RANGE_N0V256_P0V256   	(22)//range:+/-±0.256V
#define CHANNEL_RANGE_N0V512_P0V512   	(23)//range:+/-0.512V
#define CHANNEL_RANGE_N0V625_P0V625     (84)//range:+/-0.625V
#define CHANNEL_RANGE_N25mV_P25mV     	(86)//range:+/-25mV
#define CHANNEL_RANGE_N50mV_P50mV     	(87)//range:+/-50mV
#define CHANNEL_RANGE_N0V1_P0V1     	(88)//range:+/-0.1V
#define CHANNEL_RANGE_N0V25_P0V25     	(89)//range:+/-0.25V
#define CHANNEL_RANGE_N0V5_P0V5         (90)//range:+/-0.5V
#define CHANNEL_RANGE_N1V_P1V   		(24)//range:+/-1V
#define CHANNEL_RANGE_N1V024_P1V024   	(25)//range:+/-1.024V
#define CHANNEL_RANGE_N1V25_P1V25   	(26)//range:+/-1.25V
#define CHANNEL_RANGE_N2V_P2V   		(28)//range:+/-2V
#define CHANNEL_RANGE_N2V048_P2V048   	(29)//range:+/-2.048V
#define CHANNEL_RANGE_N2V5_P2V5 		(32)//range:+/-2.5V
#define CHANNEL_RANGE_N4V096_P4V096   	(33)//range:+/-4.096V
#define CHANNEL_RANGE_N5V_P5V   		(36)//range:+/-5V
#define CHANNEL_RANGE_N10V_P10V 		(40)//range:+/-10V
#define CHANNEL_RANGE_N12V_P12V 		(41)//range:+/-12V
#define CHANNEL_RANGE_N12V5_P12V5 		(42)//range:+/-12.5V
#define CHANNEL_RANGE_0_P20mA           (64)//range:0-20mA
#define CHANNEL_RANGE_N20mA_P20mA       (65)//range:+/-20mA
#define CHANNEL_RANGE_0_P40mA           (66)//range:0-40mA
#define CHANNEL_RANGE_N40mA_P40mA       (67)//range:+/-40mA
/*
*   DAC module struct and API definition
*/

/************config*********************/
#define DAC_TRIGGER_MODE_AUTO 0 //auto start
#define DAC_TRIGGER_MODE_SOFT 1 //soft trigger
#define DAC_TRIGGER_MODE_HARD 2 //hard trigger,now is not supported


typedef struct dac_wavepara{
    double *buf;
    unsigned int buflen;
    unsigned int cycles;
    double frequency; //sample rate
    unsigned char trigger_mode;
}dac_wavepara_t;
DLL_API int DLL_CALL libdaq_dac_set_value(const char* device_name, const char *module_name, double value);

DLL_API int DLL_CALL libdaq_dac_set_wavepara(const char* device_name, const char *module_name, dac_wavepara_t *wavepara);
DLL_API int DLL_CALL libdaq_dac_set_wavepara_ex(const char* device_name, const char *module_name,
        double *buf, unsigned int buflen, unsigned int cycles, double frequency,unsigned char trigger_mode);
DLL_API int DLL_CALL libdaq_dac_start(const char* device_name,const char *module_name);
DLL_API int DLL_CALL libdaq_dac_stop(const char* device_name,const char *module_name);

/*
 * ADC module struct and API definition
 */
//adc channel config
#define ADC_SAMPLE_MODE_SEQUENCE    (0x00) //sequence mode
#define ADC_SAMPLE_MODE_GROUP       (0x01) //group mode
#define ADC_SAMPLE_MODE_SYNC        (0x02) //sync mode
//adc channel parameter
#define ADC_CHANNEL_COUPLE_MODE_DC (0x00) //DC couple
#define ADC_CHANNEL_COUPLE_MODE_AC (0x01) //AC couple

#define ADC_CHANNEL_REFGND_RSE  (0x00)//referenced single ended
#define ADC_CHANNEL_REFGND_NRSE (0x01)//non referenced single ended
#define ADC_CHANNEL_REFGND_DIFF (0x02)//differencial

// those difinition will be obsoleted in future
#define ADC_CHANNEL_RANGE_0_P1V     		(4)//range:0-1V
#define ADC_CHANNEL_RANGE_0_P2V     		(8)//range:0-2V
#define ADC_CHANNEL_RANGE_0_P2V5    		(12)//range:0-2.5V
#define ADC_CHANNEL_RANGE_0_P5V     		(16)//range:0-5V
#define ADC_CHANNEL_RANGE_0_P10V    		(20)//range:0-10V
#define ADC_CHANNEL_RANGE_N0V256_P0V256   	(22)//range:+/-0.256V
#define ADC_CHANNEL_RANGE_N0V512_P0V512   	(23)//range:+/-0.512V
#define ADC_CHANNEL_RANGE_N1V_P1V   		(24)//range:+/-1V
#define ADC_CHANNEL_RANGE_N1V024_P1V024   	(25)//range:+/-1.024V
#define ADC_CHANNEL_RANGE_N2V_P2V   		(28)//range:+/-2V
#define ADC_CHANNEL_RANGE_N2V048_P2V048   	(29)//range:+/-2.048V
#define ADC_CHANNEL_RANGE_N2V5_P2V5 		(32)//range:+/-2.5V
#define ADC_CHANNEL_RANGE_N4V096_P4V096   	(33)//range:+/-4.096V
#define ADC_CHANNEL_RANGE_N5V_P5V   		(36)//range:+/-5V
#define ADC_CHANNEL_RANGE_N10V_P10V 		(40)//range:+/-10V

/******ADC_TRIG_SRC_SW config************/
//trigger_source
#define ADC_TRIG_SRC_SW  0x00
#define ADC_TRIG_SRC_HWD 0x01
#define ADC_TRIG_SRC_HWA 0x02
#define ADC_TRIG_SRC_ANY 0x03
//trigger_type
#define ADC_TRIG_TYPE_EDGE 0x00
#define ADC_TRIG_TYPE_LEVEL  0x01
//trigger_edge
#define ADC_TRIG_EDGE_RISE  0x00
#define ADC_TRIG_EDGE_FALL  0x01
#define ADC_TRIG_EDGE_BOTH  0x02
//trigger_level
#define ADC_TRIG_LEVEL_HIGH 0x00
#define ADC_TRIG_LEVEL_LOW  0x01

typedef struct adc_channelpara
{
    unsigned char channel;
    unsigned char range;
    unsigned char couplemode; //input couple mode ,AC or DC
    unsigned char refground;  //referenced single ended,NON referenced single ended,differencial
}adc_channelpara_t;

typedef struct adc_samplepara
{
    unsigned char *channel_list;
    unsigned char channel_count;
    unsigned char sample_mode;  //sequence mode, group mode,sync mode
    unsigned int frequency;   //sample rate (Hz)
    unsigned int cycles;      //0:continues
    unsigned int group_interval;  //only used in group mode(us)
}adc_samplepara_t;

DLL_API int DLL_CALL libdaq_adc_config_channel(const char* device_name,const char* module_name, adc_channelpara_t *adc_channelpara,unsigned char channel_count);
DLL_API int DLL_CALL libdaq_adc_config_channel_ex(const char* device_name,const char *module_name,
                                               unsigned char channel,unsigned char range,
                                               unsigned char couplemode,unsigned char refground);
DLL_API int DLL_CALL libdaq_adc_calibrate_channel(const char* deviceName, const char* module_name,
                                                  unsigned char channel, double gain, double offset);
DLL_API int DLL_CALL libdaq_adc_singleSample(const char* device_name, const char* module_name, unsigned char* channellist, unsigned int listlen,double* resultbuf);

DLL_API int DLL_CALL libdaq_adc_set_sample_parameter(const char* device_name,const char* module_name,
                                                     adc_samplepara_t *samplepara);
DLL_API int DLL_CALL libdaq_adc_set_sample_parameter_ex(const char* device_name,const char* module_name,
        unsigned char *channel_list,unsigned char channel_count, unsigned char sample_mode,
        unsigned int frequency,unsigned int cycles,unsigned int group_interval);
DLL_API int DLL_CALL libdaq_adc_clear_buffer(const char* device_name,const char* module_name);
DLL_API int DLL_CALL libdaq_adc_read_analog(const char* device_name,const char* module_name,
                                            double *databuf, unsigned int buflen, unsigned int* actuallen);//get input data
DLL_API int DLL_CALL libdaq_adc_read_analog_sync(const char* deviceName,const char *module_name,
                                                 double *databuf, unsigned int buflen, unsigned int *actuallen, int timeout);

DLL_API int DLL_CALL libdaq_adc_send_trigger(const char* device_name,const char* module_name);
DLL_API int DLL_CALL libdaq_adc_stop(const char* device_name,const char* module_name);

DLL_API int DLL_CALL libdaq_adc_start_task(const char* device_name, const char* module_name);
DLL_API int DLL_CALL libdaq_adc_stop_task(const char* device_name, const char* module_name);

DLL_API int DLL_CALL libdaq_adc_config_triggerSrc(const char* deviceName,const char *module_name,unsigned char trigger_source,unsigned char trigger_channel,
                                                  unsigned char trigger_type,unsigned char trigger_edge,unsigned char trigger_level,int trigger_delay);
DLL_API int DLL_CALL libdaq_adc_select_triggerSrc(const char* deviceName,const char *module_name,unsigned char trigger_source);

DLL_API int DLL_CALL libdaq_adc_extractChannelData(double *all_databuf, unsigned int all_datalen, unsigned char ch_listlen,unsigned char ch_index, double* ch_databuf);
DLL_API int DLL_CALL libdaq_adc_set_realtime(const char* deviceName, const char* module_name,int realtime_ms);

/*
 * counter module struct and API definition
 */
// counter mode edge
#define COUNTER_EDGE_RISING         0x00  //count mode rising edge
#define COUNTER_EDGE_FALLING        0x01  //count mode falling edge
#define COUNTER_EDGE_BOTH           0x02  //count mode both edge
// counter mode
#define COUNTER_MODE_COUNTER        0x10  //measure counter, measure rising/falling/both edge
#define COUNTER_MODE_PERIOD         0x11  //measure period,  measure time bettwen two rising edge
#define COUNTER_MODE_PWM            0x12  //measure pwm,     measure time bettwen two rising edge,and time bettwen rising 
#define COUNTER_MODE_ENCODER_x2     0x15  //measure encoder ,x2 mode, counts up/down on TI1FP1 edge depending on TI2FP2 level
#define COUNTER_MODE_ENCODER_x4     0x16  //measure encoder ,x4 mode, counts up/down on both TI1FP1 and TI2FP2 edges depending on the level of the other input edge and falling edge

typedef struct counter_data
{
    unsigned int period;/* unit:ns 1ns~4.29s:1Ghz~0.23hz */
    unsigned int pulse; /* unit:ns (pulse<=period) */
    int count;
}counter_data_t;

DLL_API int DLL_CALL libdaq_counter_config(const char* device_name,const char* module_name, unsigned char mode,unsigned char edge,unsigned int period);
DLL_API int DLL_CALL libdaq_counter_start(const char* device_name,const char* module_name);
DLL_API int DLL_CALL libdaq_counter_stop(const char* device_name,const char* module_name);
DLL_API int DLL_CALL libdaq_counter_get(const char* device_name,const char* module_name,counter_data_t *value);
DLL_API int DLL_CALL libdaq_counter_get_ex(const char* device_name,const char* module_name,unsigned int *period,unsigned int *pulse,int *count);
DLL_API int DLL_CALL libdaq_counter_clear(const char* device_name,const char* module_name);

/*
 * pwm module struct and API definition
 */
DLL_API int DLL_CALL libdaq_pwm_config(const char* device_name,const char *module_name);
DLL_API int DLL_CALL libdaq_pwm_set(const char* device_name,const char* module_name,unsigned int period,unsigned int pulse,unsigned int count);
DLL_API int DLL_CALL libdaq_pwm_start(const char* device_name,const char *module_name);
DLL_API int DLL_CALL libdaq_pwm_stop(const char* device_name,const char *module_name);

#ifdef __cplusplus
}
#endif

#endif // DAQLIB_H
