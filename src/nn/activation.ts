import type {NewTensor} from "./tensor_new.ts";

export type ActivationCallback = (tensor: NewTensor) => NewTensor;

export const NoneActivation: ActivationCallback = (tensor: NewTensor) => {
    return tensor;
}

export const TanhActivation: ActivationCallback = (tensor: NewTensor) => {
    return tensor.tanh();
}

export const ReluActivation: ActivationCallback = (tensor: NewTensor) => {
    return tensor.relu();
}