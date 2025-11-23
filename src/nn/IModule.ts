import type {Tensor} from "./tensor.ts";

export interface IModule {
    forward(input: Tensor): Tensor;
}