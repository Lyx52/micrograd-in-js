import {Value} from "./value.ts";
import {getTotalElements} from "./utils.ts";
import type {TensorDimension} from "./tensor.ts";
import {RandomGenerator} from "./random.ts";
export type NumberArray = (NumberArray|number)[];

const getDimensions = (dimension: NumberArray|number, dims: number[] = []): number[] => {
    if (Array.isArray(dimension)) {
        const dim = dimension.length;
        dims.push(dim);
        getDimensions(dimension[0], dims);
    }

    return dims;
}

const flatten = (dimension: NumberArray): number[] => {
    if (Array.isArray(dimension)) {
        return dimension.flatMap(v => flatten(v as NumberArray[]));
    }

    return dimension;
}

const getLength = (dimension: NumberArray, target: number, current: number = 0): number => {
    if (target === current) {
        return dimension.length;
    }

    return getLength(dimension, target, current + 1);
}

const makeView = (elements: NumberArray, dims: number[], currentIndex: number = 0, currentLevel: number = 0): [NumberArray, number] => {
    const levels = dims.length - 1;
    const buffer: NumberArray = [];
    if (currentLevel === levels) {
        return [elements.slice(currentIndex, currentIndex + dims[currentLevel]), currentIndex + dims[currentLevel]];
    }

    for (let i = 0; i < dims[currentLevel]; i++) {
        const [view, index] = makeView(elements, dims, currentIndex,  currentLevel + 1);
        currentIndex = index;
        buffer.push(view);
    }

    return [buffer, currentIndex];
}

export class NewTensor {
    private backing: number[];
    private dimensions: number[];
    private elements: number;
    public Grad: number;
    constructor(...dims: number[]) {
        if (dims.length === 0) {
            throw new Error('Tensor must be atleast one dimensional');
        }

        this.dimensions = [...dims];
        this.elements = getTotalElements(...dims);
        this.backing = new Array(this.elements).fill(0);
        this.Grad = 0;
    }

    public static from(...values: NumberArray) {
        const dims = getDimensions(values);
        const tensor = new NewTensor(...dims);
        tensor.backing = flatten(values);

        return tensor;
    }

    public isScalar() {
        return this.dimensions.length === 1 && this.dimensions[0] === 1;
    }

    public scalar(): number {
        if (!this.isScalar()) {
            throw new Error('Tensor must be a scalar value!');
        }

        return this.backing[0];
    }

    public item(): number[] {
        return this.backing;
    }

    public clone() {
        const tensor = new NewTensor(...this.dimensions)
        tensor.backing = this.item();
        tensor.Grad = this.Grad;

        return tensor;
    }

    public length(dimension: number = 0): number {
        return getLength(this.backing, dimension, this.elements);
    }

    public repr(): NumberArray {
        return this.view(...this.dimensions);
    }

    public flatten(): NewTensor {
        return NewTensor.from(...this.item());
    }

    public view(...dims: number[]): NumberArray {
        const viewElements = getTotalElements(...dims);
        if (viewElements !== this.elements) {
            throw new Error(`View has ${viewElements} elements while expected ${this.elements} elements`);
        }

        const [result, _] = makeView(this.backing, dims);

        return result;
    }

    public equalDimensions(other: NewTensor): boolean {
        if (other.dimensions.length !== this.dimensions.length) {
            return false;
        }

        for (let i = 0; i < this.dimensions.length; i++) {
            if (this.dimensions[i] !== other.dimensions[i]) {
                return false;
            }
        }

        return true;
    }

    public static randn(...dims: number[]): NewTensor {
        const tensor = new  NewTensor(...dims);

        for (let i = 0; i < tensor.backing.length; i++) {
            tensor.backing[i] = RandomGenerator.random();
        }

        return tensor;
    }
}