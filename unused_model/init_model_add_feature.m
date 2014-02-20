function model = init_model_add_feature(vocab)

% Init the logistic regression model with features added
% Reference: liblinear package and fsvd function
% 
% Usage:
%
%   model = init_model_add_feature(vocab)
% 
% Load the logistic regression model from data folder,  and add some
% necessary dependency(liblinear). 
% The function should take a vocabulary struct as input
%
% EXAMPLES:
%
% >> model = init_model_add_feature(vocab);
%
% Init the logistic regression model with features added
addpath(genpath('./liblinear'));
run ./liblinear/matlab/make;

load ../data/add_feature_punc_model;

model.lg = model_lg_add_punc;

