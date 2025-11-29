import {benchmarkEnd, benchmarkStart, getTotalElements} from "./utils.ts";
import {RandomGenerator} from "./random.ts";
import {Flatten} from "./layers/flatten.ts";
import {Value} from "../old/value.ts";
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

export type BackwardFunction = (parent: NewTensor) => void;

const EmptyBackward: BackwardFunction = (parent: NewTensor) => {
    if (parent.children.length > 0) {
        throw new Error('Not implemented');
    }
}

const MulBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('MulBackward')
    const first = parent.children[0];
    const second = parent.children[1];
    if (parent.isScalar()) {
        for (let i = 0; i < first.gradients.length; i++) {
            first.gradients[i] += second.backing[i] * parent.gradients[0];
            second.gradients[i] += first.backing[i] * parent.gradients[0];
        }
    } else {
        if (!first.equalDimensions(parent) || !second.equalDimensions(parent)) {
            throw new Error('Expected equal tensors dimensions to parent (SumBackward)')
        }

        for (let i = 0; i < parent.gradients.length; i++) {
            first.gradients[i] += second.backing[i] * parent.gradients[i];
            second.gradients[i] += first.backing[i] * parent.gradients[i];
        }
    }

    benchmarkEnd('MulBackward')
}

const DivBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('DivBackward')
    const first = parent.children[0];
    const second = parent.children[1];
    const y = second.backing;
    const x = first.backing;

    if (parent.isScalar()) {
        for (let i = 0; i < first.gradients.length; i++) {
            first.gradients[i] += (1 / y[i]) * parent.gradients[0];
            second.gradients[i] += (-x[i] / (y[i] * y[i])) * parent.gradients[0];
        }
    } else {
        if (!first.equalDimensions(parent) || !second.equalDimensions(parent)) {
            throw new Error('Expected equal tensors dimensions to parent (SumBackward)')
        }

        for (let i = 0; i < parent.gradients.length; i++) {
            first.gradients[i] += (1 / y[i]) * parent.gradients[i];
            second.gradients[i] += (-x[i] / (y[i] * y[i])) * parent.gradients[i];
        }
    }
    benchmarkEnd('DivBackward')
}

const PowBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('PowBackward')
    const first = parent.children[0];
    const second = parent.children[1];

    const x = first.backing;
    const z = first.backing.map((v, i) => Math.pow(v, second.backing[i]));

    if (parent.isScalar()) {
        for (let i = 0; i < first.gradients.length; i++) {
            first.gradients[i] += second.backing[i] * Math.pow(x[i], (second.backing[i]- 1)) * parent.gradients[0];
            second.gradients[i] += z[i] * Math.log(x[i]) * parent.gradients[0];
        }
    } else {
        if (!first.equalDimensions(parent) || !second.equalDimensions(parent)) {
            throw new Error('Expected equal tensors dimensions to parent (SumBackward)')
        }

        for (let i = 0; i < parent.gradients.length; i++) {
            first.gradients[i] += second.backing[i] * Math.pow(x[i], (second.backing[i]- 1)) * parent.gradients[i];
            second.gradients[i] += z[i] * Math.log(x[i]) * parent.gradients[i];
        }
    }

    benchmarkEnd('PowBackward')
}

const AddBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('AddBackward')
    if (parent.isScalar()) {
        for (const child of parent.children) {
            for (let i = 0; i < child.gradients.length; i++) {
                child.gradients[i] += parent.gradients[0];
            }
        }
    } else {
        parent.children.forEach(child => {
            if (!child.equalDimensions(parent)) {
                throw new Error('Expected equal tensors dimensions to parent (SumBackward)')
            }
        });

        for (let i = 0; i < parent.gradients.length; i++) {
            for (const child of parent.children) {
                child.gradients[i] += parent.gradients[i];
            }
        }
    }
    benchmarkEnd('AddBackward')
}

const SumBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('SumBackward')
    const child = parent.children[0];
    if (parent.isScalar()) {
        for (let i = 0; i < child.gradients.length; i++) {
            child.gradients[i] += parent.gradients[0];
        }
    } else {
        if (!child.equalDimensions(parent)) {
            throw new Error('Expected equal tensors dimensions to parent (SumBackward)')
        }

        for (let i = 0; i < parent.gradients.length; i++) {
            child.gradients[i] += parent.gradients[i];
        }
    }

    benchmarkEnd('SumBackward')
}

const ReluBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('ReluBackward')
    const child = parent.children[0];
    if (parent.isScalar()) {
        for (let i = 0; i < child.gradients.length; i++) {
            child.gradients[i] += Math.max(0, parent.scalar()) * parent.gradients[0];
        }
    } else {
        if (!child.equalDimensions(parent)) {
            throw new Error('Expected equal tensors dimensions to parent (ReluBackward)')
        }

        for (let i = 0; i < parent.gradients.length; i++) {
            child.gradients[i] += Math.max(0, parent.backing[i]) * parent.gradients[i];
        }
    }
    benchmarkEnd('ReluBackward')
}

const TanhBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('TanhBackward')
    const child = parent.children[0];
    const coshX = parent.backing.map(v =>  Math.cosh(v));
    if (parent.isScalar()) {
        for (let i = 0; i < child.gradients.length; i++) {
            if (Math.abs(coshX[i]) < Number.EPSILON) {
                console.warn("Unknown area, cosh(x) result for tanh is close to 0?")
                child.gradients[i] += 0;
            } else {
                child.gradients[i] += (1 / (coshX[i] * coshX[i])) * parent.gradients[0];
            }
        }
    } else {
        if (!child.equalDimensions(parent)) {
            throw new Error('Expected equal tensors dimensions to parent (TanhBackward)')
        }

        for (let i = 0; i < parent.gradients.length; i++) {
            if (Math.abs(coshX[i]) < Number.EPSILON) {
                console.warn("Unknown area, cosh(x) result for tanh is close to 0?")
                child.gradients[i] += 0;
            } else {
                child.gradients[i] += (1 / (coshX[i] * coshX[i])) * parent.gradients[i];
            }
        }
    }


    benchmarkEnd('TanhBackward')
}

const FromTensorsBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('FromTensorsBackward')
    if (parent.children.length === parent.gradients.length) {
        for (let i = 0; i < parent.children.length; i++) {
            const child = parent.children[i];
            if (!child.isScalar()) {
                throw new Error('What to do in this situation? child not scalar');
            }

            child.gradients[0] += parent.gradients[i];
        }

        return;
    }

    if (parent.children.length <= 0) {
        return;
    }


    for (let i = 0; i < parent.children.length; i++) {
        const child = parent.children[i];
        if (parent.gradients.length > child.gradients.length) {
            const count = parent.gradients.length / parent.children.length;
            child.gradients = parent.gradients.slice(i * count, i * count + count);
        } else {
            throw new Error('FromTensorsBackward');
        }
    }
    benchmarkEnd('FromTensorsBackward')
}

const FlattenBackward: BackwardFunction = (parent: NewTensor) => {
    benchmarkStart('FlattenBackward')
    const child = parent.children[0];

    for (let i = 0; i < parent.gradients.length; i++) {
        child.gradients[i] += parent.gradients[i];
    }

    benchmarkEnd('FlattenBackward')
}

export class NewTensor {
    public _backward: BackwardFunction = EmptyBackward;
    private dimensions: number[];
    private elements: number;
    public children: NewTensor[];
    public gradients: number[];
    public backing: number[];
    constructor(dims: number[], children: NewTensor[] = []) {
        if (dims.length === 0) {
            throw new Error('Tensor must be atleast one dimensional');
        }

        this.dimensions = [...dims];
        this.elements = getTotalElements(...dims);
        this.backing = new Array(this.elements).fill(0);
        this.gradients = new Array(this.backing.length).fill(0);
        this.children = children;
    }

    public static from(values: NumberArray) {
        const dims = getDimensions(values);
        const tensor = new NewTensor(dims);
        tensor.backing = flatten(values);

        return tensor;
    }

    public static fromTensors(tensors: NewTensor[]): NewTensor {
        const data = [];
        for (const tensor of tensors) {
            data.push(...tensor.backing);
        }

        return NewTensor.from(data)
            .setChildren(tensors)
            .setBackward(FromTensorsBackward);
    }

    public setChildren(children: NewTensor[]): NewTensor {
        this.children = children;

        return this;
    }

    public setDimension(dimensions: number[]): NewTensor {
        this.dimensions = dimensions;

        return this;
    }

    public setBackward(backward: BackwardFunction): NewTensor {
        this._backward = backward;

        return this;
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
        const tensor = new NewTensor(this.dimensions)
        tensor.backing = this.backing;
        tensor.gradients = this.gradients;
        tensor.children = this.children;
        tensor._backward = this._backward;
        return tensor;
    }

    public length(dimension: number = 0): number {
        return getLength(this.backing, dimension, this.elements);
    }

    public repr(): NumberArray {
        return this.view(...this.dimensions);
    }

    public flatten(): NewTensor {
        const tensor = new NewTensor([this.backing.length], [this]);
        tensor.backing = this.backing;
        tensor.gradients = this.gradients;
        tensor._backward = FlattenBackward;
        return tensor;
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
        const tensor = new  NewTensor(dims);

        for (let i = 0; i < tensor.backing.length; i++) {
            tensor.backing[i] = RandomGenerator.random();
        }

        return tensor;
    }

    public grads() {
        const gradients = [];
        gradients.push(...this.gradients);
        gradients.push(...this.children.map(v => v.grads()));
        return gradients;
    }

    public backward() {
        if (!this.isScalar()) {
            throw new Error("Backwards can only be used on scalar tensors");
        }

        this.gradients = new Array(this.backing.length).fill(1);

        const nodes = this.topological().reverse();
        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i];

            if (node._backward) {
                node._backward(node);
                continue;
            }

            console.error("Backwards not implemented on node ", node);
            throw new Error("Backwards not implemented on node ");
        }
    }

    public expand(dims: number[]): NewTensor {
        if (!this.isScalar()) {
            throw new Error("Can only expand scalar tensor");
        }

        this.dimensions = dims;
        const total = getTotalElements(...dims);
        const value = this.backing.shift();
        const grad = this.gradients.shift();

        this.backing = new Array(total).fill(value);
        this.gradients = new Array(total).fill(grad);
        this.elements = total;
        return this;
    }

    public topological(tensors: NewTensor[] = []): NewTensor[] {
        if (tensors.includes(this)) {
            return tensors;
        }

        for (const child of this.children) {
            child.topological(tensors);
        }

        tensors.push(this);
        return tensors;
    }

    public op(other: number|NewTensor, backward: BackwardFunction, callback: (first: number, second: number) => number) {
        let otherTensor = other;
        if (otherTensor instanceof NewTensor) {
            if (!this.equalDimensions(otherTensor)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${otherTensor.dimensions.join(', ')}]`);
            }
        } else {
            otherTensor = new NewTensor(this.dimensions);
            otherTensor.backing = new Array(this.backing.length).fill(other);
        }
        const tensor = NewTensor.from(this.item()
            .map((v, i) => callback(v, otherTensor.backing[i]))
        );

        return tensor
            .setDimension(this.dimensions)
            .setChildren([this, otherTensor])
            .setBackward(backward);
    }

    zerograd(): NewTensor {
        this.gradients = new Array(this.backing.length).fill(0);
        for (const child of this.children) {
            child.zerograd();
        }
        return this;
    }

    public mul(other: number|NewTensor) {
        return this.op(other, MulBackward, (first, second) => first * second);
    }

    public add(other: number|NewTensor) {
        return this.op(other, AddBackward, (first, second) => first + second);
    }

    public div(other: number|NewTensor) {
        return this.op(other, DivBackward, (first, second) => first / second);
    }

    public pow(other: number|NewTensor) {
        return this.op(other, PowBackward, (first, second) => Math.pow(first, second));
    }

    public mse(expected: number|NewTensor): NewTensor {
        if (!(expected instanceof NewTensor)) {
            expected = NewTensor.from([expected]).expand(this.dimensions);
        }

        return expected.clone().sub(this).pow(2);
    }

    public relu() {
        for (let i = 0; i < this.backing.length; i++) {
            this.backing[i] = Math.max(0, this.backing[i]);
        }

        const tensor = new NewTensor(this.dimensions, [this]);
        tensor._backward = ReluBackward;
        return tensor;
    }

    public sum(): NewTensor {
        let sum = 0;
        for (let i = 0; i < this.backing.length; i++) {
            sum += this.backing[i];
        }

        const tensor = new NewTensor([1], [this]);
        tensor.backing = [sum];
        tensor._backward = SumBackward;
        return tensor;
    }

    public negate() {
        return this.mul(-1);
    }

    public tanh() {
        const tensor = new NewTensor(this.dimensions, [this]);
        for (let i = 0; i < this.backing.length; i++) {
            tensor.backing[i] = Math.tanh(this.backing[i]);
        }

        tensor._backward = TanhBackward;
        return tensor;
    }

    public softmax(): NewTensor {
        const sum = this.sum().scalar();
        return this.div(sum);
    }

    public sub(other: number|NewTensor) {
        if (other instanceof NewTensor) {
            return this.negate().add(other);
        }

        return this.add(other * -1);
    }
}