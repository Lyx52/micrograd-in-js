<template>
  <BRow cols="1">
    <BCol class="mb-4 border-1">
      <BCard
          class="list-group-item cursor-move border-1 shadow-sm user-select-none"
      >
        <h5>Dataset</h5>
        <div v-if="!useMinstDataset" class="d-flex gap-3 mb-3" v-for="input in values">
          <BInputGroup
              prepend="Inputs"
          >
            <BFormTextarea v-model="input.x" rows="1"/>
          </BInputGroup>
          <BInputGroup
              prepend="Outputs"
          >
            <BFormTextarea v-model="input.y" rows="1"/>
          </BInputGroup>

          <BButton class="my-auto" variant="danger" @click="() => removeData(input.id)">
            X
          </BButton>
        </div>
        <div class="d-flex gap-3">
          <BButton v-if="!useMinstDataset" class="my-auto" variant="success" @click="addData">
            +
          </BButton>
          <BFormCheckbox
              class="my-auto"
              v-model="useMinstDataset"
          >
            Use minst dataset
          </BFormCheckbox>
        </div>

      </BCard>
    </BCol>
    <BCol class="mb-4 border-1">
      <BCard
          class="list-group-item cursor-move border-1 shadow-sm user-select-none"
      >
        <h5>Training</h5>
        <LossChart v-if="training" :loss="lossData"/>
        <div class="d-flex justify-content-between gap-2">
          <BButton class="text-nowrap" variant="success" @click="runNetwork" v-if="!training">
            Start training
          </BButton>
          <BButton class="text-nowrap" variant="danger" @click="stopNetwork" v-if="training">
            Stop training
          </BButton>
          <BFormInput
              class="mt-auto"
              v-model="seed"
              type="number"
              placeholder="Seed"
          />
          <BFormInput
              class="mt-auto"
              v-model="learningRate"
              type="number"
              placeholder="Learning rate"
          />
          <BFormInput
              class="mt-auto"
              v-model="epochs"
              type="number"
              placeholder="Epochs"
          />
          <BFormInput
              class="mt-auto"
              v-model="lossEveryN"
              type="number"
              placeholder="Loss every N epochs"
          />
        </div>
      </BCard>
    </BCol>
    <BCol class="mb-4 border-1" v-if="!useMinstDataset">
      <BCard
          class="list-group-item cursor-move border-1 shadow-sm user-select-none"
      >
        <h5>Testing</h5>
        <div class="d-flex mb-3">
          <code>
            {{ testingResult }}
          </code>
        </div>
        <div class="d-flex justify-content-between gap-2">
          <BInputGroup>
            <template #prepend>
              <BButton class="text-nowrap" variant="success" @click="testNetwork">
                Execute
              </BButton>
            </template>
            <BFormTextarea v-model="testingInput" rows="1"/>
          </BInputGroup>
        </div>
      </BCard>
    </BCol>
    <BCol class="mb-4 border-1">
      <BCard
          class="list-group-item cursor-move border-1 shadow-sm user-select-none"
      >
        <h5>Add layer</h5>
        <div class="d-flex mb-3 gap-3">
          <BButton variant="info" @click="useExampleMLP">
            Use example MLP
          </BButton>
          <BButton variant="info" @click="useExampleMinst">
            Use example Minst
          </BButton>
        </div>
        <div class="d-flex justify-content-between gap-2">
          <select class="form-select" v-model="layer.type">
            <option v-for="item in layerOptions" :key="item.value" :value="item.value">
              {{ item.text }}
            </option>
          </select>
          <BFormInput
              v-model="layer.inputs"
              type="number"
              placeholder="Inputs"
          />
          <BFormInput
              v-model="layer.outputs"
              type="number"
              placeholder="Outputs"
          />
          <BButton variant="success" @click="addLayer">
            +
          </BButton>
        </div>
      </BCard>
    </BCol>
    <BCol>
      <div class="d-flex gap-3">
        <h5>Network graph</h5>
        <BFormCheckbox
            v-model="renderNetwork"
            @change="list = [...list]"
        >
          Show
        </BFormCheckbox>
      </div>
      <NetworkRenderer v-if="renderNetwork" :layers="list"/>
    </BCol>
    <BCol v-if="list.length > 0">
      <h5>Network layer configuration</h5>
      <div class="flex justify-center">
        <draggable class="dragArea list-group w-full gap-3" v-model="list" :sort="true">
          <BCard
              class="list-group-item cursor-move border-1 shadow-sm user-select-none"
              v-for="(element, index) in list"
              :key="element.id"
          >
            <div class="d-flex justify-content-between gap-2">
              <select class="form-select" v-model="element.type">
                <option v-for="item in layerOptions" :key="item.value" :value="item.value">
                  {{ item.text }}
                </option>
              </select>
              <BFormInput
                  v-model="element.inputs"
                  type="number"
                  placeholder="Inputs"
                  :class="{
                    'text-bg-danger': validateLayer(element),
                  }"
              />
              <BFormInput
                  v-model="element.outputs"
                  type="number"
                  placeholder="Outputs"
              />
              <BButton variant="danger" @click="() => removeLayer(element.id)">
                X
              </BButton>
            </div>
          </BCard>
        </draggable>
      </div>
    </BCol>
  </BRow>
</template>

<script lang="ts">
import {defineComponent, ref} from 'vue'
import {VueDraggableNext} from "vue-draggable-next";
import {
  BButton,
  BCard,
  BCol,
  BFormInput,
  BFormTextarea,
  BListGroup,
  BListGroupItem,
  BRow,
  BInputGroup,
  BFormCheckbox
} from "bootstrap-vue-next";
import {NetworkLayer} from "../render/NetworkLayer.ts";
import {getLayerOptions, LayerType} from "../render/LayerTypes.ts";
import {LinearModule} from "../nn/module/linear_module.ts";
import NetworkRenderer from "./NetworkRenderer.vue";
import LossChart from "./LossChart.vue";
import {createWorker} from "../worker_context.ts";

interface IValues {
  x: string;
  y: string;
  id: number;
}

interface INetworkBuilder {
  list: NetworkLayer[];
  layer: NetworkLayer;
  id: number;
  network: {
    module?: LinearModule;
  }
  values: IValues[]
}

const loss = ref<number[]>([]);
const instance = createWorker();
const worker = instance.comlink;

instance.worker.addEventListener("message", (event: MessageEvent) => {
  if (event?.data?.type !== 'LOSS') {
    return;
  }
  loss.value = event.data.data as number[];
});

// @ts-ignore
export default defineComponent<INetworkBuilder>({
  methods: {
    getLayerOptions,
    removeLayer(id: number) {
      this.list = this.list.filter((item: NetworkLayer) => item.id !== id);
    },
    removeData(id: number) {
      this.values = this.values.filter((item: IValues) => item.id !== id);
    },
    addLayer() {
      const layer = new NetworkLayer(this.id++, this.layer.type, Number(this.layer.inputs), Number(this.layer.outputs));
      this.list.push(layer);
      this.layer.type = LayerType.ReLu;
      this.inputs = Number(this.outputs);
      this.outputs = 10;
    },
    addData() {
      this.values.push({
        id: this.dataId++,
        x: '',
        y: '',
      })
    },
    async testNetwork() {
      if (this.list.length <= 0) {
        return;
      }

      const xs = this.testingInput.trim().split(' ').map(v => Number(v)).filter(v => !isNaN(v));

      if (xs.length !== Number(this.list[0].inputs)) {
        return;
      }

      const result = await worker.testModule(xs);
      this.testingResult = JSON.stringify(result);
    },
    async stopNetwork() {
      await worker.cancelTraining();
      this.training = false;
    },
    async runNetwork() {
      const seed = Number(this.seed) >= 0 ? Number(this.seed) : Math.random() * 100;
      await worker.createModule(this.list.map(v => new NetworkLayer(v.id, v.type, Number(v.inputs), Number(v.outputs))), seed)
      if (this.useMinstDataset) {
        worker.startTrainingMinst(Number(this.epochs), Number(this.learningRate), Number(this.lossEveryN));
      } else {
        const [xs, ys] = this.getData();
        worker.startTraining(Number(this.epochs), Number(this.learningRate), Number(this.lossEveryN), xs ?? [], ys ?? []);
      }

      this.training = true;
    },
    validateLayer(layer) {
      if (this.list.length > 1) {
        const indexOfLayer = this.list.indexOf(layer);
        if (indexOfLayer < 1) return false;
        return Number(this.list[indexOfLayer - 1].outputs) !== Number(this.list[indexOfLayer].inputs);
      }
      return false;
    },
    getData() {
      const xdata = [];
      const ydata = [];
      if (this.list.length <= 0) {
        return [];
      }

      for (let i = 0; i < this.values.length; i++) {
        const xs = this.values[i].x.trim().split(' ').map(v => Number(v)).filter(v => !isNaN(v));
        const ys = this.values[i].y.trim().split(' ').map(v => Number(v)).filter(v => !isNaN(v));
        if (xs.length !== Number(this.list[0].inputs)) {
          continue;
        }

        if (ys.length !== Number(this.list[this.list.length - 1].outputs)) {
          continue;
        }

        xdata.push(xs);
        ydata.push(ys);
      }

      return [xdata, ydata];
    },
    useExampleMLP() {
      this.list = [
        new NetworkLayer(0, LayerType.Linear, 3, 4),
        new NetworkLayer(1, LayerType.Linear, 4, 4),
        new NetworkLayer(2, LayerType.Linear, 4, 1),
      ];

      this.id = 3;
      this.renderNetwork = true;
      this.useMinstDataset = false;
      this.values = [
        {
          x: '2.0 3.0 -1.0',
          y: '1.0',
          id: 0,
        },
        {
          x: '3.0 -1.0 0.5',
          y: '1.0',
          id: 1,
        },
        {
          x: '0.5 1.0 1.0',
          y: '-1.0',
          id: 2,
        },
        {
          x: '1.0 1.0 -1.0',
          y: '1.0',
          id: 3,
        },
      ];
    },
    useExampleMinst() {
      this.list = [
        new NetworkLayer(0, LayerType.Flatten, 28 * 28, 28 * 28),
        new NetworkLayer(2, LayerType.Linear, 28 * 28, 128),
        new NetworkLayer(2, LayerType.Linear, 128, 64),
        new NetworkLayer(2, LayerType.Linear, 64, 32),
        new NetworkLayer(3, LayerType.Linear, 32, 10),
        new NetworkLayer(4, LayerType.Softmax, 10, 10),
      ];

      this.id = 5;
      this.renderNetwork = false;
      this.useMinstDataset = true;
      this.values = [];
    }
  },
  components: {
    BFormCheckbox,
    LossChart,
    BInputGroup,
    BFormTextarea,
    NetworkRenderer,
    BRow,
    BCol,
    BButton,
    BFormInput,
    BCard,
    BListGroup,
    BListGroupItem,
    draggable: VueDraggableNext,
  },
  data() {
    return {
      id: 3,
      dataId: 4,
      layerOptions: getLayerOptions(),
      training: false,
      useMinstDataset: false,
      renderNetwork: true,
      layer: {
        type: LayerType.ReLu,
        inputs: 10,
        outputs: 1,
      },
      epochs: 20,
      learningRate: 0.001,
      lossEveryN: 3,
      seed: -1,
      lossData: loss,
      list: [],
      testingInput: '',
      testingResult: '',
      values: [
        {
          x: '2.0 3.0 -1.0',
          y: '1.0',
          id: 0,
        },
        {
          x: '3.0 -1.0 0.5',
          y: '1.0',
          id: 1,
        },
        {
          x: '0.5 1.0 1.0',
          y: '-1.0',
          id: 2,
        },
        {
          x: '1.0 1.0 -1.0',
          y: '1.0',
          id: 3,
        },
      ]
    }
  },
})
</script>

<style scoped></style>