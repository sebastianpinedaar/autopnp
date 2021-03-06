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
 * $Id: coreLoopConfig.h 4236 2013-07-17 09:56:15Z rupanov $
 */

/**
 * \file
 *         Configuration callbacks for RTE Scheduler. These functions need to be declared externally.
 */

/**
 * \ingroup core_loop RTE Scheduler
 * @{
 *
 */

#ifndef __RTE_CONFIG_H
#define __RTE_CONFIG_H

#include "xme/defines.h"

XME_EXTERN_C_BEGIN

/**
 * \brief Called back on schedule activation.
 *
 * \returns Operation status: XME_STATUS_SUCCESS in case of success, a relevant error code in case of error.
 */
extern xme_status_t
xme_core_loop_ActivateScheduleCallback( void );

/**
 * \brief Called back to perform module registration. Is the place where components and functions have to be defined
 *          by filling relevant descriptors.
 *
 * \returns Operation status: XME_STATUS_SUCCESS in case of success, a relevant error code in case of error.
 */
extern xme_status_t
xme_core_loop_RegisterModulesCallback( void );

/**
 * \brief A callback function to be defined by the system configurator to define the chunks array.
 *
 * \returns Operation status: XME_STATUS_SUCCESS in case of success, a relevant error code in case of error.
 */
extern xme_status_t
xme_core_loop_CreateChunksCallback( void );

XME_EXTERN_C_END

#endif

/**
 * @}
 */
