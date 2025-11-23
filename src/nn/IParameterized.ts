import type {Value} from "./value.ts";

export interface IParameterized {
    parameters(): Value[];
    zerograd();
}