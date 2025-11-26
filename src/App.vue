<script setup lang="ts">
import {BCol, BContainer, BRow} from "bootstrap-vue-next";
import NetworkBuilder from "./components/NetworkBuilder.vue";
</script>

<template>
  <BContainer class="bv-example-row">
    <BRow cols="1">
      <BCol>
        <NetworkBuilder />
      </BCol>
    </BRow>
  </BContainer>
</template>

<style scoped>

</style>
