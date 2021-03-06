/*
 * Copyright (c) 2011-2013, fortiss GmbH.
 * Licensed under the Apache License, Version 2.0.
 *
 * Use, modification and distribution are subject to the terms specified
 * in the accompanying license file LICENSE.txt located at the root directory
 * of this software distribution. A copy is available at
 * http://chromosome.fortiss.org/.
 *
 * This file is part of CHROMOSOME.
 *
 * $Id: funcFunction.c 4948 2013-09-04 08:25:51Z ruiz $
 */

/**
 * \file
 *         Source file for function func in component wrapperTest.
 *
 * \author
 *         This file has been generated by the CHROMOSOME Modeling Tool (XMT)
 *         (fortiss GmbH).
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "xme/adv/test/wrapperTest/include/funcFunction.h"

#include "xme/adv/test/wrapperTest/include/funcFunctionWrapper.h"
#include "xme/adv/test/wrapperTest/include/wrapperTestComponentWrapper.h"

// PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_C_INCLUDES) ENABLED START

XME_EXTERN_C_BEGIN

/******************************************************************************/
/***   Global variables                                                     ***/
/******************************************************************************/
uint8_t xme_adv_wrapperTest_mode = 0;

// PROTECTED REGION END

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/
/**
 * \brief  Variable holding the value of the required output port 'out0'.
 *
 * \details If necessary initialize this in the init function.
 *          The value of this variable will be written to the port at the end of
 *          the step function.
 */
static xme_core_topic_login_loginResponse_t
portOut0Data;

/**
 * \brief  Variable holding the value of the optional output port 'out1'.
 *
 * \details If necessary initialize this in the init function.
 *          The value of this variable will be written to the port at the end of
 *          the step function.
 */
static xme_core_topic_login_pnpLoginResponse_t
portOut1Data;

// PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_C_STATICVARIABLES) ENABLED START
    
// PROTECTED REGION END

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/
xme_status_t
xme_adv_wrapperTest_funcFunction_init
(
    void* param
)
{
    // PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_INITIALIZE_C) ENABLED START
    
    // TODO: Auto-generated stub
    
    XME_UNUSED_PARAMETER(param);
    
    return XME_STATUS_SUCCESS;
    
    // PROTECTED REGION END
}

void
xme_adv_wrapperTest_funcFunction_step
(
    void* param
)
{
    xme_status_t status[2];
    
    xme_core_topic_login_loginRequest_t portIn0Data; // Required port.
    xme_core_topic_login_pnpLoginRequest_t portIn1Data; // Optional port.
    xme_core_topic_login_loginResponse_t* portOut0DataPtr = &portOut0Data;
    xme_core_topic_login_pnpLoginResponse_t* portOut1DataPtr = &portOut1Data;
    
    status[0] = xme_adv_wrapperTest_wrapperTestComponentWrapper_readPortIn0(&portIn0Data);
    status[1] = xme_adv_wrapperTest_wrapperTestComponentWrapper_readPortIn1(&portIn1Data);
    
    {
        // PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_STEP_C) ENABLED START
    
        // TODO: Auto-generated stub
    
        XME_UNUSED_PARAMETER(param);
        XME_UNUSED_PARAMETER(status);
        
        switch (xme_adv_wrapperTest_mode)
        {
            case 0:
                {
                    portOut1DataPtr = NULL;
                }
                break;
            case 1:
                {
                    portOut0DataPtr = NULL;
                }
                break;
            default:
                XME_LOG
                (
                    XME_LOG_ERROR,
                    "xme_adv_wrapperTest_funcFunction_step: Unknown mode %d",
                    xme_adv_wrapperTest_mode
                );
        }
        
        // PROTECTED REGION END
    }
    
    status[0] = xme_adv_wrapperTest_wrapperTestComponentWrapper_writePortOut0(portOut0DataPtr);
    status[1] = xme_adv_wrapperTest_wrapperTestComponentWrapper_writePortOut1(portOut1DataPtr);
    
    {
        // PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_STEP_2_C) ENABLED START
    
        // Do nothing
    
        // PROTECTED REGION END
    }
}

void
xme_adv_wrapperTest_funcFunction_fini(void)
{
    // PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_TERMINATE_C) ENABLED START
    
    // Do nothing
    
    // PROTECTED REGION END
}

// PROTECTED REGION ID(XME_ADV_WRAPPERTEST_FUNCFUNCTION_IMPLEMENTATION_C) ENABLED START

// PROTECTED REGION END

XME_EXTERN_C_END
