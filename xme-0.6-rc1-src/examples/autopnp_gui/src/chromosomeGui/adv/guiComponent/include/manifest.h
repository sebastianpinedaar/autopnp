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
 *         Defines function for creation of component type manifest
 *         for 'guiComponent'
 *
 * \author
 *         This file has been generated by the CHROMOSOME Modeling Tool (XMT)
 *         (fortiss GmbH).
 */
 
#ifndef CHROMOSOMEGUI_ADV_GUICOMPONENT_MANIFEST_H
#define CHROMOSOMEGUI_ADV_GUICOMPONENT_MANIFEST_H

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "xme/core/manifestTypes.h"

#include "xme/hal/include/mem.h"
#include "xme/hal/include/safeString.h"

XME_EXTERN_C_BEGIN

/******************************************************************************/
/***   Prototypes                                                           ***/
/******************************************************************************/
/**
 * \brief  Create manifest for for component type 'guiComponent'.
 *         Component type id: 5001.
 *
 * \param  componentManifest Pointer to component type manifest structure that
 *         will be populated. Must not be NULL.
 *
 * \retval XME_STATUS_INVALID_PARAMETER when componentManifest is NULL.
 * \retval XME_STATUS_SUCCESS when componentManifest has been created
 *         successfully.
 */
xme_status_t
chromosomeGui_adv_guiComponent_manifest_createComponentTypeManifest
(
    xme_core_componentManifest_t* componentManifest
);

XME_EXTERN_C_END

#endif // #ifndef CHROMOSOMEGUI_ADV_GUICOMPONENT_MANIFEST_H
