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
 * $Id$
 */

/**
 * \file
 *
 * \brief  Waypoint that marshals topic data for network transportation.
 *
 * \author
 *         This file has been generated by the CHROMOSOME Modeling Tool (fortiss GmbH).
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "chromosomeGui/wp/marshaler/include/marshaler.h"
#include "chromosomeGui/topic/dictionary.h"
#include "chromosomeGui/topic/dictionaryData.h"

#include "xme/hal/include/mem.h"
#include "xme/hal/include/net.h"
#include "xme/hal/include/table.h"

#include "xme/core/dataHandler/include/dataHandler.h"
#include "xme/core/topic.h"
#include "xme/core/topicData.h"

#include <inttypes.h>

/******************************************************************************/
/***   Type definitions                                                     ***/
/******************************************************************************/
/**
 * \struct configurationItem_t
 *
 * \brief  Structure for storing a configuration for this waypoint.
 */
typedef struct
{
	xme_core_topic_t topic; ///< Topic of the data that is stored at inputPort
	xme_core_dataManager_dataPacketId_t inputPort; ///< InputPort where topic data is stored
	xme_core_dataManager_dataPacketId_t outputPort; ///< OutputPort where marshaled data should be written to
}
configurationItem_t;

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/
/**
 * \brief  Table for storing this waypoints configurations.
 */
static XME_HAL_TABLE
(
	configurationItem_t,
	configurationTable,
	XME_WP_MARSHALER_CONFIGURATIONTABLE_SIZE
);

/**
 * \brief  Constant array that contains all topics that are supported by this
 *         marshaler.
 */
static const xme_core_topic_t
supportedTopics[] =
{
	XME_CORE_TOPIC(CHROMOSOMEGUI_TOPIC_BUTTONSIGNAL),
	XME_CORE_TOPIC(CHROMOSOMEGUI_TOPIC_WRITETEXT),
	XME_CORE_TOPIC(XME_CORE_TOPIC_PNPMANAGER_RUNTIME_GRAPH_MODEL),
	XME_CORE_TOPIC(XME_CORE_TOPIC_PNPMANAGER_RUNTIME_GRAPH_MODEL2),
	XME_CORE_TOPIC(XME_CORE_TOPIC_PNPMANAGER_RUNTIME_GRAPH_MODEL3)
};

/******************************************************************************/
/***   Prototypes                                                           ***/
/******************************************************************************/
/**
 * \brief  Performs marshaling on the given topic data.
 *
 * \param  topic      Topic of the data stored at inputPort.
 * \param  inputPort  The inputPort where the data is stored that will be marshaled.
 * \param  outputPort The outputPort where the marshaled data is written to.
 *
 * \retval XME_STATUS_SUCCESS if the data was marshaled successfully.
 * \retval XME_STATUS_UNSUPPORTED if the given topic is not supported by this marshaler.
 * \retval XME_STATUS_INTERNAL_ERROR if an error occurred during reading from the inputPort.
 */
xme_status_t
doMarshaling
(
	xme_core_topic_t topic,
	xme_core_dataManager_dataPacketId_t inputPort,
	xme_core_dataManager_dataPacketId_t outputPort
);

/**
 * \brief  Performs marshaling for topic 'ButtonSignal'.
 *
 * \param  inputPort Input port where data is stored that will be marshaled.
 * \param  buffer Buffer where marshaled data will be written to.
 *
 * \retval XME_STATUS_SUCCESS if the data was marshaled successfully.
 * \retval XME_STATUS_INTERNAL_ERROR if an error occurred during reading from the inputPort.
 */
xme_status_t
doMarshalingForButtonSignal
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
);
/**
 * \brief  Performs marshaling for topic 'WriteText'.
 *
 * \param  inputPort Input port where data is stored that will be marshaled.
 * \param  buffer Buffer where marshaled data will be written to.
 *
 * \retval XME_STATUS_SUCCESS if the data was marshaled successfully.
 * \retval XME_STATUS_INTERNAL_ERROR if an error occurred during reading from the inputPort.
 */
xme_status_t
doMarshalingForWriteText
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
);
/**
 * \brief  Performs marshaling for topic 'pnpManager_runtime_graph_model'.
 *
 * \param  inputPort Input port where data is stored that will be marshaled.
 * \param  buffer Buffer where marshaled data will be written to.
 *
 * \retval XME_STATUS_SUCCESS if the data was marshaled successfully.
 * \retval XME_STATUS_INTERNAL_ERROR if an error occurred during reading from the inputPort.
 */
xme_status_t
doMarshalingForPnpManager_runtime_graph_model
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
);
/**
 * \brief  Performs marshaling for topic 'pnpManager_runtime_graph_model2'.
 *
 * \param  inputPort Input port where data is stored that will be marshaled.
 * \param  buffer Buffer where marshaled data will be written to.
 *
 * \retval XME_STATUS_SUCCESS if the data was marshaled successfully.
 * \retval XME_STATUS_INTERNAL_ERROR if an error occurred during reading from the inputPort.
 */
xme_status_t
doMarshalingForPnpManager_runtime_graph_model2
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
);
/**
 * \brief  Performs marshaling for topic 'pnpManager_runtime_graph_model3'.
 *
 * \param  inputPort Input port where data is stored that will be marshaled.
 * \param  buffer Buffer where marshaled data will be written to.
 *
 * \retval XME_STATUS_SUCCESS if the data was marshaled successfully.
 * \retval XME_STATUS_INTERNAL_ERROR if an error occurred during reading from the inputPort.
 */
xme_status_t
doMarshalingForPnpManager_runtime_graph_model3
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
);

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/
xme_status_t
chromosomeGui_wp_marshaler_init(void)
{
	XME_HAL_TABLE_INIT(configurationTable);

	return XME_STATUS_SUCCESS;
}

xme_status_t
chromosomeGui_wp_marshaler_run
(
	xme_wp_waypoint_instanceId_t instanceId
)
{
	xme_status_t status;
	configurationItem_t* configurationItem;

	configurationItem = XME_HAL_TABLE_ITEM_FROM_HANDLE
	(
		configurationTable,
		(xme_hal_table_rowHandle_t)instanceId
	);

	XME_CHECK
	(
		NULL != configurationItem,
		XME_STATUS_INVALID_HANDLE
	);
	
	// Do the marshaling for this configuration
	status = doMarshaling
	(
		configurationItem->topic,
		configurationItem->inputPort,
		configurationItem->outputPort
	);

	XME_CHECK
	(
		XME_STATUS_SUCCESS == status, 
		XME_STATUS_INTERNAL_ERROR
	);

	return XME_STATUS_SUCCESS;
}

xme_status_t
chromosomeGui_wp_marshaler_addConfig
(
	xme_wp_waypoint_instanceId_t* instanceId,
	xme_core_topic_t topic,
	xme_core_dataManager_dataPacketId_t inputPort,
	xme_core_dataManager_dataPacketId_t outputPort
)
{
	xme_hal_table_rowHandle_t configurationItemHandle;
	configurationItem_t* configurationItem;

	XME_CHECK
	(
		chromosomeGui_wp_marshaler_isSupported(topic),
		XME_STATUS_INVALID_PARAMETER
	);

	configurationItemHandle = XME_HAL_TABLE_ADD_ITEM(configurationTable);

	XME_CHECK
	(
		XME_HAL_TABLE_INVALID_ROW_HANDLE != configurationItemHandle,
		XME_STATUS_OUT_OF_RESOURCES
	);

	configurationItem = XME_HAL_TABLE_ITEM_FROM_HANDLE
	(
		configurationTable,
		configurationItemHandle
	);
	
	XME_ASSERT(NULL != configurationItem);

	configurationItem->topic = topic;
	configurationItem->inputPort = inputPort;
	configurationItem->outputPort = outputPort;

	// We use the row handle to identify this configuration
	*instanceId = (xme_wp_waypoint_instanceId_t)configurationItemHandle;

	return XME_STATUS_SUCCESS;
}

xme_status_t
chromosomeGui_wp_marshaler_fini(void)
{
	XME_HAL_TABLE_FINI(configurationTable);

	return XME_STATUS_SUCCESS;
}

xme_status_t
doMarshaling
(
	xme_core_topic_t topic,
	xme_core_dataManager_dataPacketId_t inputPort,
	xme_core_dataManager_dataPacketId_t outputPort
)
{
	void* buffer;
	unsigned int bufferSize;
	xme_status_t status;

	// Switch for the correct topic
	// In the respective cases we allocate a buffer with the right size for the topic and
	// call a function that performs the read from the inputPort and the actual marshaling
	if (XME_CORE_TOPIC(CHROMOSOMEGUI_TOPIC_BUTTONSIGNAL) == topic)
	{
		uint8_t marshaledData[1];
		
		buffer = marshaledData;
		bufferSize = 1;
		
		status = doMarshalingForButtonSignal
		(
			inputPort,
			(void*)marshaledData
		);
	}
	else if (XME_CORE_TOPIC(CHROMOSOMEGUI_TOPIC_WRITETEXT) == topic)
	{
		uint8_t marshaledData[1000];
		
		buffer = marshaledData;
		bufferSize = 1000;
		
		status = doMarshalingForWriteText
		(
			inputPort,
			(void*)marshaledData
		);
	}
	else if (XME_CORE_TOPIC(XME_CORE_TOPIC_PNPMANAGER_RUNTIME_GRAPH_MODEL) == topic)
	{
		uint8_t marshaledData[2962];
		
		buffer = marshaledData;
		bufferSize = 2962;
		
		status = doMarshalingForPnpManager_runtime_graph_model
		(
			inputPort,
			(void*)marshaledData
		);
	}
	else if (XME_CORE_TOPIC(XME_CORE_TOPIC_PNPMANAGER_RUNTIME_GRAPH_MODEL2) == topic)
	{
		uint8_t marshaledData[2962];
		
		buffer = marshaledData;
		bufferSize = 2962;
		
		status = doMarshalingForPnpManager_runtime_graph_model2
		(
			inputPort,
			(void*)marshaledData
		);
	}
	else if (XME_CORE_TOPIC(XME_CORE_TOPIC_PNPMANAGER_RUNTIME_GRAPH_MODEL3) == topic)
	{
		uint8_t marshaledData[2962];
		
		buffer = marshaledData;
		bufferSize = 2962;
		
		status = doMarshalingForPnpManager_runtime_graph_model3
		(
			inputPort,
			(void*)marshaledData
		);
	}
	else
	{
		XME_LOG
		(
			XME_LOG_ERROR,
			"xme_wp_marshaler_run(): Given topic with id %" PRIu64 " is not "
			"supported by this marshaler.",
			topic
		);
		return XME_STATUS_INTERNAL_ERROR;
	}

	XME_CHECK
	(
		XME_STATUS_SUCCESS == status,
		XME_STATUS_INTERNAL_ERROR
	);

	// Write marshaled data to outputPort
	status = xme_core_dataHandler_writeData
	(
		outputPort,
		buffer,
		bufferSize
	);

	XME_CHECK
	(
		XME_STATUS_SUCCESS == status,
		XME_STATUS_INTERNAL_ERROR
	);
	
	xme_core_dataHandler_completeWriteOperation(outputPort);
	xme_core_dataHandler_completeReadOperation(inputPort);

	return XME_STATUS_SUCCESS;
}

xme_status_t
doMarshalingForButtonSignal
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
)
{
	chromosomeGui_topic_ButtonSignal_t topicData;
	unsigned int topicDataSize;
	unsigned int bytesRead;
	uint8_t* bufferPtr;
	xme_status_t status;

	topicDataSize = sizeof(chromosomeGui_topic_ButtonSignal_t);

	// Read topic data
	status = xme_core_dataHandler_readData
	(
		inputPort,
		&topicData,
		topicDataSize,
		&bytesRead
	);

	XME_CHECK(status == XME_STATUS_SUCCESS, XME_STATUS_INTERNAL_ERROR);
	XME_CHECK(bytesRead == topicDataSize, XME_STATUS_INTERNAL_ERROR);

	// Marshal topic data
	bufferPtr = (uint8_t*)buffer;
	
	// char topicData.buttonPushed
	xme_hal_mem_copy(bufferPtr, &topicData.buttonPushed, sizeof(char));
	bufferPtr += sizeof(char);
	
	return XME_STATUS_SUCCESS;
}
xme_status_t
doMarshalingForWriteText
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
)
{
	chromosomeGui_topic_WriteText_t topicData;
	unsigned int topicDataSize;
	unsigned int bytesRead;
	uint8_t* bufferPtr;
	xme_status_t status;

	topicDataSize = sizeof(chromosomeGui_topic_WriteText_t);

	// Read topic data
	status = xme_core_dataHandler_readData
	(
		inputPort,
		&topicData,
		topicDataSize,
		&bytesRead
	);

	XME_CHECK(status == XME_STATUS_SUCCESS, XME_STATUS_INTERNAL_ERROR);
	XME_CHECK(bytesRead == topicDataSize, XME_STATUS_INTERNAL_ERROR);

	// Marshal topic data
	bufferPtr = (uint8_t*)buffer;
	
	// char topicData.text
	{
		uint16_t i0;
		
		for (i0 = 0; i0 < 1000; i0++)
		{
			// char topicData.text[i0]
			xme_hal_mem_copy(bufferPtr, &topicData.text[i0], sizeof(char));
			bufferPtr += sizeof(char);
		}
	}
	
	return XME_STATUS_SUCCESS;
}
xme_status_t
doMarshalingForPnpManager_runtime_graph_model
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
)
{
	xme_core_topic_pnpManager_runtime_graph_model_t topicData;
	unsigned int topicDataSize;
	unsigned int bytesRead;
	uint8_t* bufferPtr;
	xme_status_t status;

	topicDataSize = sizeof(xme_core_topic_pnpManager_runtime_graph_model_t);

	// Read topic data
	status = xme_core_dataHandler_readData
	(
		inputPort,
		&topicData,
		topicDataSize,
		&bytesRead
	);

	XME_CHECK(status == XME_STATUS_SUCCESS, XME_STATUS_INTERNAL_ERROR);
	XME_CHECK(bytesRead == topicDataSize, XME_STATUS_INTERNAL_ERROR);

	// Marshal topic data
	bufferPtr = (uint8_t*)buffer;
	
	// uint16_t topicData.nodeId
	{
		uint16_t netValue;
		
		netValue = xme_hal_net_htons((uint16_t)topicData.nodeId);
		xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint16_t));
		bufferPtr += sizeof(uint16_t);
	}
	
	// struct topicData.vertex
	{
		uint8_t i0;
		
		for (i0 = 0; i0 < 10; i0++)
		{
			// struct topicData.vertex[i0]
			{
				// uint8_t topicData.vertex[i0].vertexType
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.vertex[i0].vertexType;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// char topicData.vertex[i0].vertexData
				{
					uint16_t i1;
					
					for (i1 = 0; i1 < 256; i1++)
					{
						// char topicData.vertex[i0].vertexData[i1]
						xme_hal_mem_copy(bufferPtr, &topicData.vertex[i0].vertexData[i1], sizeof(char));
						bufferPtr += sizeof(char);
					}
				}
				
				// uint32_t topicData.vertex[i0].instanceId
				{
					uint32_t netValue;
					
					netValue = xme_hal_net_htonl((uint32_t)topicData.vertex[i0].instanceId);
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint32_t));
					bufferPtr += sizeof(uint32_t);
				}
			}
		}
	}
	
	// struct topicData.edge
	{
		uint8_t i0;
		
		for (i0 = 0; i0 < 10; i0++)
		{
			// struct topicData.edge[i0]
			{
				// int8_t topicData.edge[i0].srcVertexIndex
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].srcVertexIndex;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// int8_t topicData.edge[i0].sinkVertexIndex
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].sinkVertexIndex;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// uint8_t topicData.edge[i0].edgeType
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].edgeType;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// char topicData.edge[i0].edgeData
				{
					uint8_t i1;
					
					for (i1 = 0; i1 < 32; i1++)
					{
						// char topicData.edge[i0].edgeData[i1]
						xme_hal_mem_copy(bufferPtr, &topicData.edge[i0].edgeData[i1], sizeof(char));
						bufferPtr += sizeof(char);
					}
				}
			}
		}
	}
	
	return XME_STATUS_SUCCESS;
}
xme_status_t
doMarshalingForPnpManager_runtime_graph_model2
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
)
{
	xme_core_topic_pnpManager_runtime_graph_model2_t topicData;
	unsigned int topicDataSize;
	unsigned int bytesRead;
	uint8_t* bufferPtr;
	xme_status_t status;

	topicDataSize = sizeof(xme_core_topic_pnpManager_runtime_graph_model2_t);

	// Read topic data
	status = xme_core_dataHandler_readData
	(
		inputPort,
		&topicData,
		topicDataSize,
		&bytesRead
	);

	XME_CHECK(status == XME_STATUS_SUCCESS, XME_STATUS_INTERNAL_ERROR);
	XME_CHECK(bytesRead == topicDataSize, XME_STATUS_INTERNAL_ERROR);

	// Marshal topic data
	bufferPtr = (uint8_t*)buffer;
	
	// uint16_t topicData.nodeId
	{
		uint16_t netValue;
		
		netValue = xme_hal_net_htons((uint16_t)topicData.nodeId);
		xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint16_t));
		bufferPtr += sizeof(uint16_t);
	}
	
	// struct topicData.vertex
	{
		uint8_t i0;
		
		for (i0 = 0; i0 < 10; i0++)
		{
			// struct topicData.vertex[i0]
			{
				// uint8_t topicData.vertex[i0].vertexType
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.vertex[i0].vertexType;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// char topicData.vertex[i0].vertexData
				{
					uint16_t i1;
					
					for (i1 = 0; i1 < 256; i1++)
					{
						// char topicData.vertex[i0].vertexData[i1]
						xme_hal_mem_copy(bufferPtr, &topicData.vertex[i0].vertexData[i1], sizeof(char));
						bufferPtr += sizeof(char);
					}
				}
				
				// uint32_t topicData.vertex[i0].instanceId
				{
					uint32_t netValue;
					
					netValue = xme_hal_net_htonl((uint32_t)topicData.vertex[i0].instanceId);
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint32_t));
					bufferPtr += sizeof(uint32_t);
				}
			}
		}
	}
	
	// struct topicData.edge
	{
		uint8_t i0;
		
		for (i0 = 0; i0 < 10; i0++)
		{
			// struct topicData.edge[i0]
			{
				// int8_t topicData.edge[i0].srcVertexIndex
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].srcVertexIndex;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// int8_t topicData.edge[i0].sinkVertexIndex
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].sinkVertexIndex;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// uint8_t topicData.edge[i0].edgeType
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].edgeType;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// char topicData.edge[i0].edgeData
				{
					uint8_t i1;
					
					for (i1 = 0; i1 < 32; i1++)
					{
						// char topicData.edge[i0].edgeData[i1]
						xme_hal_mem_copy(bufferPtr, &topicData.edge[i0].edgeData[i1], sizeof(char));
						bufferPtr += sizeof(char);
					}
				}
			}
		}
	}
	
	return XME_STATUS_SUCCESS;
}
xme_status_t
doMarshalingForPnpManager_runtime_graph_model3
(
	xme_core_dataManager_dataPacketId_t inputPort,
	void* buffer
)
{
	xme_core_topic_pnpManager_runtime_graph_model3_t topicData;
	unsigned int topicDataSize;
	unsigned int bytesRead;
	uint8_t* bufferPtr;
	xme_status_t status;

	topicDataSize = sizeof(xme_core_topic_pnpManager_runtime_graph_model3_t);

	// Read topic data
	status = xme_core_dataHandler_readData
	(
		inputPort,
		&topicData,
		topicDataSize,
		&bytesRead
	);

	XME_CHECK(status == XME_STATUS_SUCCESS, XME_STATUS_INTERNAL_ERROR);
	XME_CHECK(bytesRead == topicDataSize, XME_STATUS_INTERNAL_ERROR);

	// Marshal topic data
	bufferPtr = (uint8_t*)buffer;
	
	// uint16_t topicData.nodeId
	{
		uint16_t netValue;
		
		netValue = xme_hal_net_htons((uint16_t)topicData.nodeId);
		xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint16_t));
		bufferPtr += sizeof(uint16_t);
	}
	
	// struct topicData.vertex
	{
		uint8_t i0;
		
		for (i0 = 0; i0 < 10; i0++)
		{
			// struct topicData.vertex[i0]
			{
				// uint8_t topicData.vertex[i0].vertexType
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.vertex[i0].vertexType;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// char topicData.vertex[i0].vertexData
				{
					uint16_t i1;
					
					for (i1 = 0; i1 < 256; i1++)
					{
						// char topicData.vertex[i0].vertexData[i1]
						xme_hal_mem_copy(bufferPtr, &topicData.vertex[i0].vertexData[i1], sizeof(char));
						bufferPtr += sizeof(char);
					}
				}
				
				// uint32_t topicData.vertex[i0].instanceId
				{
					uint32_t netValue;
					
					netValue = xme_hal_net_htonl((uint32_t)topicData.vertex[i0].instanceId);
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint32_t));
					bufferPtr += sizeof(uint32_t);
				}
			}
		}
	}
	
	// struct topicData.edge
	{
		uint8_t i0;
		
		for (i0 = 0; i0 < 10; i0++)
		{
			// struct topicData.edge[i0]
			{
				// int8_t topicData.edge[i0].srcVertexIndex
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].srcVertexIndex;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// int8_t topicData.edge[i0].sinkVertexIndex
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].sinkVertexIndex;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// uint8_t topicData.edge[i0].edgeType
				{
					uint8_t netValue;
					
					netValue = (uint8_t)topicData.edge[i0].edgeType;
					xme_hal_mem_copy(bufferPtr, &netValue, sizeof(uint8_t));
					bufferPtr += sizeof(uint8_t);
				}
				
				// char topicData.edge[i0].edgeData
				{
					uint8_t i1;
					
					for (i1 = 0; i1 < 32; i1++)
					{
						// char topicData.edge[i0].edgeData[i1]
						xme_hal_mem_copy(bufferPtr, &topicData.edge[i0].edgeData[i1], sizeof(char));
						bufferPtr += sizeof(char);
					}
				}
			}
		}
	}
	
	return XME_STATUS_SUCCESS;
}

bool
chromosomeGui_wp_marshaler_isSupported
(
	xme_core_topic_t topic
)
{
	uint64_t i;
	size_t supportTopicsLength;
	
	supportTopicsLength = sizeof(supportedTopics) / sizeof(supportedTopics[0]);
	
	for (i = 0; i < supportTopicsLength; i++)
	{
		if (topic == supportedTopics[i])
		{
			return true;
		}
	}
	
	return false;
}
