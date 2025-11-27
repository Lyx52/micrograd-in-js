<script lang="ts">
import {defineComponent} from 'vue'
import {NetworkLayer} from "../render/NetworkLayer.ts";
import {Network} from "vis-network";
import {DataSet} from "vis-data";
import {v4 as uuid} from "uuid";
export default defineComponent({
  name: "NetworkRenderer",
  props: {
    layers: Array<NetworkLayer>
  },
  methods: {
    renderGraph(layers) {
      console.log(layers);
      const createLayer = (layer) => {
        
      }

      const data = {
        nodes: new DataSet([
          { id: 1, label: "Node 1" },
          { id: 2, label: "Node 2" },
          { id: 3, label: "Node 3" },
          { id: 4, label: "Node 4" },
          { id: 5, label: "Node 5" },
        ]),
        edges: new DataSet([
          { from: 1, to: 3 },
          { from: 1, to: 2 },
          { from: 2, to: 4 },
          { from: 2, to: 5 },
          { from: 3, to: 3 },
        ]),
      };

      const renderContainer = document.querySelector('#renderContainer');
      const network = new Network(renderContainer, data, {});
    }
  },
  watch: {
    layers: {
      handler(newLayers) {
        this.renderGraph(newLayers);
      },
      deep: true
    }
  }
})
</script>

<template>
  <div class="d-flex flex-column">
    {{ JSON.stringify(layers) }}
    <div id="renderContainer">

    </div>
  </div>
</template>

<style scoped>

</style>