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
 * $Id: loginManagerComponentWrapper.c 4961 2013-09-04 12:26:40Z ruiz $
 */

/**
 * \file
 *         Component wrapper - implements interface of a component
 *              to the data handler
 *
 * \author
 *         This file has been generated by the CHROMOSOME Modeling Tool (XMT)
 *         (fortiss GmbH).
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "xme/core/login/include-gen/loginManagerComponentWrapper.h"

/******************************************************************************/
/***   Type definitions                                                     ***/
/******************************************************************************/
/**
 * \brief  Structure for storing information about the input ports.
 */
typedef struct
{
    xme_core_dataManager_dataPacketId_t dataPacketId; ///< The data packet id.
    xme_core_component_portStatus_t status; ///< Status of the port. Denotes whether the port has been read and completeReadOperation should be called.
} inputPort_t;

/**
 * \brief  Structure for storing information about the output ports.
 */
typedef struct
{
        xme_core_dataManager_dataPacketId_t dataPacketId; ///< The data packet id.
        xme_core_component_portStatus_t status; ///< Status of the port. Denotes whether the port contains a valid value and completeWriteOperation should be called.
} outputPort_t;

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/
/**
 * \brief  Array storing information about the input ports.
 */
static inputPort_t inputPorts[] = {
    {XME_CORE_DATAMANAGER_DATAPACKETID_INVALID, XME_CORE_COMPONENT_PORTSTATUS_INVALID},
    {XME_CORE_DATAMANAGER_DATAPACKETID_INVALID, XME_CORE_COMPONENT_PORTSTATUS_INVALID}
};

/**
 * \brief  Array storing information about the output ports.
 */
static outputPort_t outputPorts[] = {
    {XME_CORE_DATAMANAGER_DATAPACKETID_INVALID, XME_CORE_COMPONENT_PORTSTATUS_INVALID},
    {XME_CORE_DATAMANAGER_DATAPACKETID_INVALID, XME_CORE_COMPONENT_PORTSTATUS_INVALID}
};

/**
 * \brief  Size of inputPorts array.
 */
static const uint8_t inputPortCount = sizeof(inputPorts) / sizeof(inputPorts[0]);

/**
 * \brief  Size of outputPorts array.
 */
static const uint8_t outputPortCount = sizeof(outputPorts) / sizeof(outputPorts[0]);

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/
xme_status_t
xme_core_login_loginManagerComponentWrapper_receivePort
(
    xme_core_dataManager_dataPacketId_t dataPacketId,
    xme_core_login_loginManagerComponentWrapper_internalPortId_t componentInternalPortId
)
{
    XME_CHECK
    (
        (inputPortCount + outputPortCount) > (uint8_t)componentInternalPortId, 
        XME_STATUS_INVALID_PARAMETER
    );
    
    if ((int32_t)componentInternalPortId < inputPortCount)
    {
        inputPorts[componentInternalPortId].dataPacketId = dataPacketId;
        inputPorts[componentInternalPortId].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
    }
    else
    {
        uint32_t outputPortIndex = (uint32_t)componentInternalPortId - inputPortCount;
        outputPorts[outputPortIndex].dataPacketId = dataPacketId;
        outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
    }
    
    return XME_STATUS_SUCCESS;
}

void
xme_core_login_loginManagerComponentWrapper_completeReadOperations(void)
{
    uint8_t inputPortIndex;
    xme_status_t status;
    
    for (inputPortIndex = 0; inputPortIndex < inputPortCount; inputPortIndex++)
    {
        if (XME_CORE_COMPONENT_PORTSTATUS_VALID == inputPorts[inputPortIndex].status)
        {
            status = xme_core_dataHandler_completeReadOperation(inputPorts[inputPortIndex].dataPacketId);
            inputPorts[inputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
            if (XME_STATUS_SUCCESS != status)
            {
                XME_LOG
                (
                    XME_LOG_ERROR,
                    "[loginManagerComponentWrapper] CompleteReadOperation for port (interalPortId: %d, dataPacketId: %d) returned error code %d\n",
                    inputPortIndex,
                    inputPorts[inputPortIndex].dataPacketId,
                    status
                );
            }
        }
    }
}

void
xme_core_login_loginManagerComponentWrapper_completeWriteOperations(void)
{
    uint8_t outputPortIndex;
    xme_status_t status;
    
    for (outputPortIndex = 0; outputPortIndex < outputPortCount; outputPortIndex++)
    {
        if (XME_CORE_COMPONENT_PORTSTATUS_VALID == outputPorts[outputPortIndex].status)
        {
            status = xme_core_dataHandler_completeWriteOperation(outputPorts[outputPortIndex].dataPacketId);
            outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
            if (XME_STATUS_SUCCESS != status)
            {
                XME_LOG
                (
                    XME_LOG_ERROR,
                    "[loginManagerComponentWrapper] CompleteWriteOperation for port (interalPortId: %d, dataPacketId: %d) returned error code %d\n",
                    inputPortCount + outputPortIndex,
                    outputPorts[outputPortIndex].dataPacketId,
                    status
                );
            }
        }
    }
}

xme_status_t
xme_core_login_loginManagerComponentWrapper_readPortInLoginRequest
(
    xme_core_topic_login_loginRequest_t* data
)
{
    uint8_t inputPortIndex;
    unsigned int bytesRead;
    xme_status_t returnValue;
    
    inputPortIndex = (uint8_t) XME_CORE_LOGIN_LOGINMANAGERCOMPONENTWRAPPER_PORT_INLOGINREQUEST;
    
    returnValue = xme_core_dataHandler_readData
    (
        inputPorts[inputPortIndex].dataPacketId,
        (void*) data,
        sizeof(xme_core_topic_login_loginRequest_t),
        &bytesRead
    );
    
    if (XME_STATUS_SUCCESS == returnValue)
    {
        inputPorts[inputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_VALID;
    }
    else
    {
        inputPorts[inputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
    }
    
    return returnValue;
}

xme_status_t
xme_core_login_loginManagerComponentWrapper_readPortInPnPLoginResponse
(
    xme_core_topic_login_pnpLoginResponse_t* data
)
{
    uint8_t inputPortIndex;
    unsigned int bytesRead;
    xme_status_t returnValue;
    
    inputPortIndex = (uint8_t) XME_CORE_LOGIN_LOGINMANAGERCOMPONENTWRAPPER_PORT_INPNPLOGINRESPONSE;
    
    returnValue = xme_core_dataHandler_readData
    (
        inputPorts[inputPortIndex].dataPacketId,
        (void*) data,
        sizeof(xme_core_topic_login_pnpLoginResponse_t),
        &bytesRead
    );
    
    if (XME_STATUS_SUCCESS == returnValue)
    {
        inputPorts[inputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_VALID;
    }
    else
    {
        inputPorts[inputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
    }
    
    return returnValue;
}

xme_status_t
xme_core_login_loginManagerComponentWrapper_writePortOutPnPLoginRequest
(
    xme_core_topic_login_pnpLoginRequest_t* data
)
{
    uint8_t outputPortIndex;
    xme_status_t returnValue;
    
    outputPortIndex = ((uint8_t)XME_CORE_LOGIN_LOGINMANAGERCOMPONENTWRAPPER_PORT_OUTPNPLOGINREQUEST) - inputPortCount;
    
    XME_CHECK_REC
    (
        NULL != data,
        XME_STATUS_SUCCESS,
        {
            outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
        }
    );
    
    returnValue = xme_core_dataHandler_writeData
    (
        outputPorts[outputPortIndex].dataPacketId,
        (void*) data,
        sizeof(xme_core_topic_login_pnpLoginRequest_t)
    );
    
    if (XME_STATUS_SUCCESS == returnValue)
    {
        outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_VALID;
    }
    else
    {
        outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
        
        XME_LOG
        (
            XME_LOG_ERROR,
            "%s:%d xme_core_login_loginManagerComponentWrapper_writePortOutPnPLoginRequest: Error when writing to port. Data handler returned error code %d.",
            __FILE__,
            __LINE__,
            returnValue
        );
    }
    
    return returnValue;
}

xme_status_t
xme_core_login_loginManagerComponentWrapper_writePortOutLoginResponse
(
    xme_core_topic_login_loginResponse_t* data
)
{
    uint8_t outputPortIndex;
    xme_status_t returnValue;
    
    outputPortIndex = ((uint8_t)XME_CORE_LOGIN_LOGINMANAGERCOMPONENTWRAPPER_PORT_OUTLOGINRESPONSE) - inputPortCount;
    
    XME_CHECK_REC
    (
        NULL != data,
        XME_STATUS_SUCCESS,
        {
            outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
        }
    );
    
    returnValue = xme_core_dataHandler_writeData
    (
        outputPorts[outputPortIndex].dataPacketId,
        (void*) data,
        sizeof(xme_core_topic_login_loginResponse_t)
    );
    
    if (XME_STATUS_SUCCESS == returnValue)
    {
        outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_VALID;
    }
    else
    {
        outputPorts[outputPortIndex].status = XME_CORE_COMPONENT_PORTSTATUS_INVALID;
        
        XME_LOG
        (
            XME_LOG_ERROR,
            "%s:%d xme_core_login_loginManagerComponentWrapper_writePortOutLoginResponse: Error when writing to port. Data handler returned error code %d.",
            __FILE__,
            __LINE__,
            returnValue
        );
    }
    
    return returnValue;
}

