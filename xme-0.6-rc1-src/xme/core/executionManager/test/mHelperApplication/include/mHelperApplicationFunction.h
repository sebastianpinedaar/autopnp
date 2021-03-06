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
 * $Id: mHelperApplicationFunction.h 3545 2013-05-28 18:30:08Z rupanov $
 */

/**
 * \file
 *         Header file for function mHelperApplication in component mHelperApplication.
 *
 * \author
 *         This file has been generated by the CHROMOSOME Modeling Tool (XMT)
 *         (fortiss GmbH).
 */

#ifndef test_mHelperApplication_mHelperApplicationFUNCTION_H
#define test_mHelperApplication_mHelperApplicationFUNCTION_H

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "xme/core/component.h"

#define N_FUNCTIONS_SIMULATED 64

/******************************************************************************/
/***   Prototypes                                                           ***/
/******************************************************************************/
/**
 * \brief  Initialization of function. 
 *  
 * \details Called once before the function is executed the first time.
 *
 * \param  param Function-specific initialization parameter.
 *
 * \return XME_STATUS_SUCCESS when initialization was successful.
 */
xme_status_t
test_mHelperApplication_mHelperApplicationFunction_init
(
	void* param
);

/**
 * \brief  Executes the function one time.
 *
 * \details Input ports of the function need to be prepared, before calling this.
 *
 * \param  param Function-specific parameter
 */
void
test_mHelperApplication_mHelperApplicationFunction_step
(
	void* param
);

/**
 * \brief  Finalization function.
 *
 * \details Called after function will no longer be executed to free allocated resources.
 *          Function must not be executed after teminate has been called.
 */
void
test_mHelperApplication_mHelperApplicationFunction_fini(void);

#endif // #ifndef test_mHelperApplication_mHelperApplicationFUNCTION_H
