% |------------------------------------------------------------------------
% |               C O P Y R I G H T
% |------------------------------------------------------------------------
% | Copyright (c) 2011 by ACE Lab           All rights reserved.
% | This software is copyright protected and proprietary to ACE Lab.  
% |------------------------------------------------------------------------

%%
function blkStruct = slblocks

blkStruct.Name = 'SbPaC_Lib';

% The function that will be called when the user double-clicks on this icon
blkStruct.OpenFcn = 'SbPaC_Lib';
blkStruct.MaskDisplay = 'disp(''SbPaC_Lib'')';

Viewer(1).Library = 'simviewers';
Viewer(1).Name    = 'Simulink';

blkStruct.Viewer = Viewer;

% Define the library list for the Simulink Library browser.
% Return the name of the library model and the name for it
Browser(1).Library = 'SbPaC_Lib';
Browser(1).Name    = 'ACE Lab: SbPaC Blockset';
Browser(1).IsFlat  = 0; % Is this library "flat" (i.e. no subsystems)?

blkStruct.Browser = Browser;

% End of slblocks
