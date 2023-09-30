<template>
  <div class="Folders">
    <div class="operationBoX">
      <el-button class="operationBtn uploadBtn"type="primary" icon="el-icon-upload2" @click="uploadDialogVisible = true">Upload</el-button>
      <el-button-group class="operationBtn operationBtnGrp">
        <el-button  type="primary" icon="el-icon-video-play">Recognize</el-button>
<!--        <el-button  type="primary" icon="el-icon-circle-plus-outline">新建记忆集</el-button>-->
      </el-button-group>
      <el-input class="searchInput"
                placeholder="search"
                suffix-icon="el-icon-search"
                v-model="searchContent">
      </el-input>

      <!-- Upload Image Dialog -->
      <el-dialog :visible.sync="uploadDialogVisible" title="Upload Image" width="30%">
        <el-form :model="form">
          <el-form-item label="Picture URL">
            <el-input v-model="form.pictureFilePath" placeholder="Enter picture URL"></el-input>
          </el-form-item>
          <el-form-item label="Picture Name">
            <el-input v-model="form.pictureName" placeholder="Enter picture name"></el-input>
          </el-form-item>
        </el-form>
        <span slot="footer" class="dialog-footer">
        <el-button @click="uploadDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleUpload">Confirm</el-button>
      </span>
      </el-dialog>

    </div>
    <div class="tableBox">
      <router-view></router-view>
    </div>
  </div>

</template>

<script>

import {uploadPictures} from "@/api/pictures";

export default {
  name:'Pictures',
  data(){
    return{
      searchContent:'',
      tableData: [],
      uploadDialogVisible:false,
      form: {
        pictureName: '',
        pictureFilePath: '',
        pictureOwnerId: JSON.parse(sessionStorage.getItem('user')).userId
      }
    }
  },
  methods:{
    async handleUpload() {
      try {
        // Show $confirm message box
        await this.$confirm('Are you sure you want to upload this image?', 'Confirm Upload', {
          confirmButtonText: 'Yes',
          cancelButtonText: 'No',
          type: 'warning',
        });

        await uploadPictures(this.form)

        this.$message({
          type: 'success',
          message: 'Image uploaded successfully!',
        });

      } catch (err) {
        // User cancelled the $confirm message box
        this.$message({
          type: 'info',
          message: 'Image upload cancelled',
        });
      } finally {
        // Reset the form and close the dialog
        this.form.pictureFilePath = '';
        this.form.pictureName = '';
        this.uploadDialogVisible = false;
      }
      this.$bus.$emit('update table', 0);
    },
  },
  mounted() {
    this.$router.replace({name:'PicturesTable'})
  }
}
</script>

<style scoped>

.Folders {
  height: 100%;
}
.operationBoX {
  height: 15%;
  display: flex;
}

.tableBox {
  height: 85%;
}


.operationBtn{
  position: relative;
  margin-top: auto;
  margin-bottom: auto;
}

.uploadBtn {
  left: 50px;
}

.operationBtnGrp {
  left: 100px;
}

.searchInput {
  position: relative;
  margin-top: auto;
  margin-bottom: auto;
  width: 200px;
  left: 850px;
}


</style>