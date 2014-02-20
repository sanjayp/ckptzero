function model = init_model(vocab)

% Init the logistic regression model
% Reference: liblinear package
% 
% Usage:
%
%   model = init_model(vocab)
% 
% Load the logistic regression model from data folder and add some
% necessary dependency(liblinear). 
% The function should take a vocabulary struct as input
%
% EXAMPLES:
%
% >> model = init_model(vocab);
%
% Inint the logistic regression model

addpath(genpath('./liblinear'));
% Note: necessary to run the make file on the server
run ./liblinear/matlab/make;
load data/model_lg.mat;	% load logistic regression model file

model.lg = model_lg;	% set the model
