function model = init_model_nb(vocab)

% Init the Naive Bayes model
% Reference: liblinear package and fsvd function
% 
% Usage:
%
%   model = init_model_nb(vocab)
% 
% Load the Naive Bayes model from data folder
% The function should take a vocabulary struct as input
%
% EXAMPLES:
%
% >> model = init_model_nb(vocab);
%
% Inint the Naive Bayes model

load ../data/model_nb;

model.nb = model_nb;

